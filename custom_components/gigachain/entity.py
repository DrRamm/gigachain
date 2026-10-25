"""Conversation agent entity for the GigaChain integration."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Literal

from home_assistant_intents import get_languages
from homeassistant.components import conversation
from homeassistant.components.conversation import agent_manager
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import callback
from homeassistant.helpers import (
    area_registry as ar,
)
from homeassistant.helpers import (
    device_registry as dr,
)
from homeassistant.helpers import (
    entity_registry as er,
)
from homeassistant.helpers import (
    intent,
    llm,
    template,
)

from custom_components.voice_normalizer.const import GLOBAL_LIGHTS, LIGHTS
from custom_components.voice_normalizer.conversation import (
    _room_for_assistant,
    async_apply_result,
)
from custom_components.voice_normalizer.normalizer import normalize as normalize_local_lights

from . import geo, llm_tools
from .const import (
    CONF_ENGINE,
    CONF_LLM_HASS_API,
    CONF_PROCESS_BUILTIN_SENTENCES,
    CONF_PROMPT,
    DEFAULT_LLM_HASS_API,
    DEFAULT_PROCESS_BUILTIN_SENTENCES,
    DEFAULT_PROMPT_EN,
    DEFAULT_PROMPT_RU,
    DEFAULT_PROMPT_YANDEXGPT_EN,
    DEFAULT_PROMPT_YANDEXGPT_RU,
    DOMAIN,
    ID_GIGACHAT,
    ID_YANDEX_GPT,
    LLM_API_NONE,
)

LOGGER = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 10
LOCAL_COMMAND_HINT = re.compile(
    r"свет|освещ|ламп|подсвет|включ|выключ|заж|погас|потуш|пагас|патуш|сп[оа]койн|добр.*ноч",
    re.IGNORECASE,
)


def _log_timing(
    path: str, started: float, stages: dict[str, float], conversation_id: str | None
) -> None:
    """Log how long one conversation turn took and which path answered it."""
    LOGGER.info(
        "gigachain timing: conversation_id=%s path=%s builtin=%.3fs normalizer=%.3fs "
        "apply=%.3fs llm_data=%.3fs llm=%.3fs total=%.3fs",
        conversation_id, path,
        *(stages.get(name, 0.0) for name in
          ("builtin", "normalizer", "apply", "llm_data", "llm")),
        time.monotonic() - started,
    )

FUNCTION_CALL_MARKER = re.compile(r'\{\s*"function_call"')


class GigaChatAgent(conversation.ConversationEntity):
    """GigaChain conversation agent."""

    _attr_has_entity_name = True
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_name = entry.title

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return get_languages()

    async def async_added_to_hass(self) -> None:
        """When entity is added to hass."""
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from hass."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Process a message."""
        started = time.monotonic()
        stages: dict[str, float] = {}
        engine = self.entry.data.get(CONF_ENGINE) or ID_GIGACHAT
        llm_api_id = self._async_get_llm_api_id()

        use_builtin_sentences = self.entry.options.get(
            CONF_PROCESS_BUILTIN_SENTENCES, DEFAULT_PROCESS_BUILTIN_SENTENCES
        )
        if use_builtin_sentences:
            stage_started = time.monotonic()
            default_agent = agent_manager.async_get_agent(self.hass, None)
            # Probe the built-in agent against our own chat_log so that, when
            # it recognises the sentence, its messages (tool call, tool result
            # and speech) end up in the ChatLog we persist and stay visible to
            # the model on the next turn. When it does not recognise the
            # sentence we roll back whatever it appended (e.g. the "not
            # understood" speech) before consulting the LLM.
            content_snapshot = len(chat_log.content)
            default_agent_response = None
            try:
                default_agent_response = await default_agent._async_handle_message(
                    user_input, chat_log
                )
            except Exception:
                del chat_log.content[content_snapshot:]
                if "погод" not in (user_input.text or "").lower():
                    raise
            finally:
                stages["builtin"] = time.monotonic() - stage_started
            if (
                default_agent_response is not None
                and default_agent_response.response.intent
            ):
                intent_response = default_agent_response.response
                if intent_response.response_type == intent.IntentResponseType.ERROR:
                    # A matched sentence whose reply template failed, for example
                    # "погода через три часа" caught by HassStartTimer. Let the
                    # model answer instead of speaking the template error.
                    del chat_log.content[content_snapshot:]
                else:
                    intent_obj = intent_response.intent
                    text_l = (user_input.text or "").lower()
                    if intent_obj.intent_type == "HassStartTimer" and "погод" in text_l:
                        del chat_log.content[content_snapshot:]
                    else:
                        if (
                            intent_obj.intent_type in {"HassTurnOn", "HassTurnOff"}
                            and intent_response.response_type == intent.IntentResponseType.ACTION_DONE
                        ):
                            intent_response.speech.clear()
                        _log_timing("builtin", started, stages, chat_log.conversation_id)
                        return conversation.ConversationResult(
                            conversation_id=chat_log.conversation_id,
                            response=intent_response,
                        )
            del chat_log.content[content_snapshot:]

            if LOCAL_COMMAND_HINT.search(user_input.text or ""):
                stage_started = time.monotonic()
                available = {
                    entity_id for entity_id in (*LIGHTS, *GLOBAL_LIGHTS)
                    if (state := self.hass.states.get(entity_id)) is not None
                    and state.state not in ("unavailable", "unknown")
                }
                local_result = normalize_local_lights(
                    user_input.text,
                    default_room=_room_for_assistant(self.hass),
                    available_entities=available,
                )
                stages["normalizer"] = time.monotonic() - stage_started
                stage_started = time.monotonic()
                speech = (
                    await async_apply_result(
                        self.hass, local_result, user_input.context,
                        user_input.satellite_id,
                    )
                    if local_result.get("entities") or local_result.get("scene")
                    else None
                )
                stages["apply"] = time.monotonic() - stage_started
                clarification = local_result["clarification"]
                if speech or local_result.get("room") or clarification == "Скажи команду без отрицания или исключений.":
                    _log_timing(
                        "normalizer" if speech else "clarification",
                        started, stages, chat_log.conversation_id,
                    )
                    local_response = intent.IntentResponse(language=user_input.language)
                    local_response.async_set_speech(speech or clarification or "Уточни команду.")
                    return conversation.ConversationResult(
                        conversation_id=chat_log.conversation_id,
                        response=local_response,
                    )

        try:
            stage_started = time.monotonic()
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                llm_api_id,
                self._user_prompt(engine, user_input.language),
                user_input.extra_system_prompt,
            )
            stages["llm_data"] = time.monotonic() - stage_started
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        client = self.entry.runtime_data
        if engine == ID_YANDEX_GPT:
            llm_started = time.monotonic()
            await self._async_chat_with_prompt_functions(client, chat_log, user_input.language)
            stages["llm"] = time.monotonic() - llm_started
            path = "llm"
        else:
            path = await self._async_chat_with_tools(
                client, chat_log, user_input.language, stages
            )

        _log_timing(path, started, stages, chat_log.conversation_id)
        return conversation.async_get_result_from_chat_log(user_input, chat_log)

    def _async_get_llm_api_id(self) -> str | None:
        """Return the HA LLM API to use, or None for a plain chat."""
        engine = self.entry.data.get(CONF_ENGINE) or ID_GIGACHAT
        if engine == ID_YANDEX_GPT:
            return None
        llm_api_id = self.entry.options.get(CONF_LLM_HASS_API, DEFAULT_LLM_HASS_API)
        if not llm_api_id or llm_api_id == LLM_API_NONE:
            return None
        return llm_api_id

    def _user_prompt(self, engine: str, language: str | None) -> str | None:
        """Return the user prompt template for the request language."""
        if engine == ID_YANDEX_GPT:
            return None
        if self.entry.options.get(CONF_PROMPT):
            return self.entry.options[CONF_PROMPT]
        return (DEFAULT_PROMPT_RU if (language or "").lower().startswith("ru")
                else DEFAULT_PROMPT_EN)

    async def _async_chat_with_tools(
        self,
        client,
        chat_log: conversation.ChatLog,
        language: str | None,
        stages: dict[str, float],
    ) -> str:
        """Generate an answer, letting the model call HA tools natively."""
        tools = [llm_tools.location_tool_to_langchain()]
        if chat_log.llm_api is not None and chat_log.llm_api.tools:
            tools = llm_tools.hass_tools_to_langchain(chat_log.llm_api) + tools
        bound_client = client.bind_tools(tools)
        used_tool = False

        for _ in range(MAX_TOOL_ITERATIONS):
            messages = llm_tools.chat_log_to_langchain(chat_log.content)
            llm_started = time.monotonic()
            try:
                result = await bound_client.ainvoke(messages)
            except Exception:
                LOGGER.exception("Error invoking LLM")
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(
                        agent_id=self.entity_id,
                        content="Не удалось связаться с GigaChat.",
                    )
                )
                return "llm_tool" if used_tool else "llm"
            finally:
                stages["llm"] = stages.get("llm", 0.0) + time.monotonic() - llm_started
            content = llm_tools.ai_message_to_assistant_content(
                self.entity_id, result
            )
            if not content.tool_calls and content.content and FUNCTION_CALL_MARKER.search(
                content.content
            ):
                # The model replied with a legacy function_call JSON instead of
                # calling a tool; never show that raw JSON to the user.
                LOGGER.warning(
                    "Model emitted a function_call JSON instead of a tool call: %s",
                    content.content,
                )
                chat_log.async_add_user_content(
                    conversation.UserContent(
                        content=(
                            "Используй доступные инструменты вместо JSON и ответь "
                            "пользователю."
                        )
                    )
                )
                continue
            if content.tool_calls:
                used_tool = True
                external_calls = [
                    tool_call
                    for tool_call in content.tool_calls
                    if tool_call.tool_name == llm_tools.LOCATION_TOOL_NAME
                ]
                for tool_call in external_calls:
                    tool_call.external = True
                async for _tool_result in chat_log.async_add_assistant_content(
                    content
                ):
                    pass
                for tool_call in external_calls:
                    location = await geo.async_resolve_location(
                        self.hass,
                        tool_call.tool_args.get("name"),
                        language,
                    )
                    chat_log.async_add_assistant_content_without_tools(
                        conversation.ToolResultContent(
                            agent_id=self.entity_id,
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.tool_name,
                            tool_result=location,
                        )
                    )
                continue
            chat_log.async_add_assistant_content_without_tools(content)
            return "llm_tool" if used_tool else "llm"

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content="Извини, я не смог обработать запрос: слишком много действий.",
            )
        )
        return "llm_tool" if used_tool else "llm"

    async def _async_chat_with_prompt_functions(
        self,
        client,
        chat_log: conversation.ChatLog,
        language: str | None,
    ) -> None:
        """Legacy control mode for engines without native function calling.

        The exposed entities are listed in the system prompt and the model is
        asked to reply with a function_call JSON, which is mapped to HA intents.
        """
        is_ru = (language or "").lower().startswith("ru")
        default_prompt = (DEFAULT_PROMPT_YANDEXGPT_RU if is_ru
                          else DEFAULT_PROMPT_YANDEXGPT_EN)
        prompt = template.Template(
            self.entry.options.get(CONF_PROMPT) or default_prompt,
            self.hass,
        ).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": self._async_get_exposed_entities(),
            },
            parse_result=False,
        )
        chat_log.content[0] = conversation.SystemContent(content=prompt)

        result_text = ""
        for _ in range(MAX_TOOL_ITERATIONS):
            messages = llm_tools.chat_log_to_langchain(chat_log.content)
            try:
                result = await self.hass.async_add_executor_job(client.invoke, messages)
            except Exception:
                LOGGER.exception("Error invoking LLM")
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(
                        agent_id=self.entity_id,
                        content="Не удалось связаться с GigaChat.",
                    )
                )
                return
            raw = result.content or ""
            function_data = self._parse_function_call(raw)
            if function_data is None:
                if FUNCTION_CALL_MARKER.search(raw):
                    # The model tried to emit a function call but it was
                    # malformed (e.g. truncated by max_tokens), so the raw
                    # JSON must never be shown to the user.
                    LOGGER.warning("Malformed function call from model: %s", raw)
                    chat_log.async_add_user_content(
                        conversation.UserContent(
                            content=(
                                "Не удалось распознать команду. Ответь пользователю "
                                "обычным текстом или корректным JSON."
                            )
                        )
                    )
                    result_text = result_text or (
                        "Не удалось распознать команду. Повторите, пожалуйста."
                    )
                    continue
                chat_log.async_add_assistant_content_without_tools(
                    conversation.AssistantContent(
                        agent_id=self.entity_id, content=raw
                    )
                )
                return
            chat_log.async_add_assistant_content_without_tools(
                conversation.AssistantContent(
                    agent_id=self.entity_id, content=raw
                )
            )

            function_call = function_data.get("function_call")
            if (
                isinstance(function_call, dict)
                and function_call.get("type") == "location"
            ):
                location = await geo.async_resolve_location(
                    self.hass,
                    function_call.get("entity_id") or function_call.get("name"),
                    language,
                )
                result_text = geo.format_location_text(location, language)
                LOGGER.debug("Location result: %s", result_text)
                chat_log.async_add_user_content(
                    conversation.UserContent(content=result_text)
                )
                continue

            intent_type, slots = self._prepare_intent(function_data)
            if intent_type is None:
                chat_log.async_add_user_content(
                    conversation.UserContent(
                        content="Не удалось определить устройство, попроси пользователя уточнить запрос."
                    )
                )
                continue

            try:
                intent_response = await intent.async_handle(
                    self.hass, "conversation", intent_type, slots
                )
            except (intent.UnknownIntent, intent.InvalidSlotInfo, intent.IntentError) as err:
                LOGGER.error("Error handling intent %s: %s", intent_type, err)
                result_text = f"Ошибка выполнения: {err}"
            else:
                result_text = intent_response.speech.get("plain", {}).get(
                    "speech"
                ) or str(
                    intent_response.matched_states
                    or intent_response.unmatched_states
                    or "Готово"
                )
            LOGGER.debug("Function call result: %s", result_text)
            chat_log.async_add_user_content(
                conversation.UserContent(content=result_text)
            )

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(
                agent_id=self.entity_id,
                content=result_text or "Извини, не удалось выполнить запрос.",
            )
        )

    def _prepare_intent(self, function_data: dict) -> tuple[str | None, dict]:
        """Map a parsed function_call to an HA intent with slots."""
        function_call = function_data.get("function_call")
        if not isinstance(function_call, dict):
            return None, {}
        function_type = function_call.get("type")
        entity_id = function_call.get("entity_id")
        if not entity_id:
            return None, {}

        entity_registry = er.async_get(self.hass)
        entity = entity_registry.async_get(entity_id)
        if entity is None:
            LOGGER.warning("Unknown entity from model: %s", entity_id)
            return None, {}

        area_id = entity.area_id
        if area_id is None and entity.device_id is not None:
            device = dr.async_get(self.hass).async_get(entity.device_id)
            area_id = device.area_id if device else None
        area_registry = ar.async_get(self.hass)
        area = area_registry.async_get_area(area_id) if area_id else None
        area_value = area.name if area else area_id
        location_text = function_call.get("location") or area_value or ""
        aliases_text = ", ".join(
            alias for alias in (entity.aliases or []) if isinstance(alias, str)
        ) or (entity.name or "")

        intent_type = None
        slots = {}
        if function_type == "switch":
            action = function_call.get("action")
            if action == "turn_on":
                intent_type = "HassTurnOn"
            elif action == "turn_off":
                intent_type = "HassTurnOff"
            if intent_type:
                slots = {
                    "name": {
                        "value": entity.name or entity_id,
                        "text": aliases_text,
                    },
                    "area": {"value": area_value, "text": location_text},
                    "domain": {
                        "value": entity_id.partition(".")[0],
                        "text": entity_id.partition(".")[0],
                    },
                }
        elif function_type == "open_close":
            action = function_call.get("action")
            if entity_id.partition(".")[0] == "valve":
                # Valves have no open/close intents, they are turned on/off
                intent_type = "HassTurnOn" if action == "open" else "HassTurnOff"
                slots = {
                    "name": {
                        "value": entity.name or entity_id,
                        "text": aliases_text,
                    },
                    "area": {"value": area_value, "text": location_text},
                    "domain": {"value": "valve", "text": "valve"},
                }
            elif action == "open":
                intent_type = "HassOpenCover"
            elif action == "close":
                intent_type = "HassCloseCover"
            if intent_type in ("HassOpenCover", "HassCloseCover"):
                slots = {
                    "name": {
                        "value": entity.name or entity_id,
                        "text": aliases_text,
                    },
                    "area": {"value": area_value, "text": location_text},
                }
        elif function_type == "position":
            intent_type = "HassSetPosition"
            slots = {
                "name": {
                    "value": entity.name or entity_id,
                    "text": aliases_text,
                },
                "area": {"value": area_value, "text": location_text},
                "position": {"value": function_call.get("value")},
            }
        elif function_type == "property":
            intent_type = "HassGetState"
            slots = {
                "name": {
                    "value": entity.name or entity_id,
                    "text": aliases_text,
                },
                "area": {"value": area_value, "text": location_text},
            }
        LOGGER.debug("Prepared intent: %s, %s", intent_type, slots)
        return intent_type, slots

    @staticmethod
    def _parse_function_call(text: str) -> dict | None:
        """Extract a function_call JSON object from a model response."""
        match = FUNCTION_CALL_MARKER.search(text)
        if match is None:
            return None
        start = match.start()
        data = None
        try:
            data = json.loads(text[start:])
        except json.JSONDecodeError as err:
            try:
                data = json.loads(text[start : start + err.pos])
            except json.JSONDecodeError:
                LOGGER.warning("Failed to parse function call from: %s", text)
                return None
        if isinstance(data, dict) and isinstance(data.get("function_call"), dict):
            return data
        return None

    @callback
    def _async_get_exposed_entities(self) -> list[dict]:
        """Return entities exposed to the conversation agent."""
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        device_registry = dr.async_get(self.hass)
        area_registry = ar.async_get(self.hass)
        exposed_entities = []
        for state in states:
            entity = entity_registry.async_get(state.entity_id)
            area_id = entity.area_id if entity else None
            if area_id is None and entity is not None and entity.device_id:
                device = device_registry.async_get(entity.device_id)
                area_id = device.area_id if device else None
            area = area_registry.async_get_area(area_id) if area_id else None
            exposed_entities.append(
                {
                    "entity_id": state.entity_id,
                    "name": state.name,
                    "state": state.state,
                    "area": area.name if area else (area_id or ""),
                    "aliases": list(entity.aliases) if entity and entity.aliases else [],
                }
            )
        return exposed_entities
