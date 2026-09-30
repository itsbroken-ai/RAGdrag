import httpx
import pytest
import respx

from ragdrag.adapters.chat import ChatResponseError
from ragdrag.core.exfiltrate import extract_knowledge

TARGET = "https://target.test/chat"


@respx.mock
def test_custom_response_field_rejects_malformed_json():
    respx.post(TARGET).mock(return_value=httpx.Response(200, text="not-json"))
    with httpx.Client() as client:
        with pytest.raises(ChatResponseError):
            extract_knowledge(TARGET, client, queries=["one"], response_field="answer")


@respx.mock
def test_custom_response_field_accepts_object_value():
    respx.post(TARGET).mock(return_value=httpx.Response(200, json={"answer": "ordinary response"}))
    with httpx.Client() as client:
        assert extract_knowledge(TARGET, client, queries=["one"], response_field="answer") == []
