import importlib.util
import os
from pathlib import Path
from unittest.mock import patch, Mock

from ask_sdk_model import (
    RequestEnvelope,
    LaunchRequest,
    Session,
    User,
    IntentRequest,
    Intent,
    Slot,
)
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.attributes_manager import AttributesManager

# Ensure API key exists before module import
os.environ.setdefault('OPENAI_API_KEY', 'test')

# Load lambda_function module from file path
module_path = Path(__file__).resolve().parents[1] / 'lambda' / 'lambda_function.py'
spec = importlib.util.spec_from_file_location('lambda_function', module_path)
lambda_fn = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = lambda_fn
spec.loader.exec_module(lambda_fn)


def build_launch_input():
    lr = LaunchRequest(request_id='req')
    user = User(user_id='user')
    session = Session(new=True, session_id='session', user=user, attributes={})
    envelope = RequestEnvelope(request=lr, session=session)
    attr = AttributesManager(envelope)
    return HandlerInput(request_envelope=envelope, attributes_manager=attr)


def build_intent_input(query):
    slot = Slot(name='query', value=query)
    intent = Intent(name='GptQueryIntent', slots={'query': slot})
    intent_request = IntentRequest(request_id='req', intent=intent)
    user = User(user_id='user')
    session = Session(new=True, session_id='session', user=user, attributes={'chat_history': []})
    envelope = RequestEnvelope(request=intent_request, session=session)
    attr = AttributesManager(envelope)
    return HandlerInput(request_envelope=envelope, attributes_manager=attr)


def test_launch_request_handler_sets_chat_history():
    handler_input = build_launch_input()
    handler = lambda_fn.LaunchRequestHandler()
    response = handler.handle(handler_input)

    assert handler_input.attributes_manager.session_attributes['chat_history'] == []
    assert response.output_speech.ssml == '<speak>Chat G.P.T. mode activated</speak>'
    assert response.reprompt.output_speech.ssml == '<speak>Chat G.P.T. mode activated</speak>'
    assert response.should_end_session is False


@patch('lambda_function.requests.post')
def test_gpt_query_intent_handler_uses_openai(mock_post):
    handler_input = build_intent_input('hello')
    mock_response = Mock(ok=True)
    mock_response.json.return_value = {
        'choices': [{'message': {'content': 'Hi there'}}]
    }
    mock_post.return_value = mock_response

    handler = lambda_fn.GptQueryIntentHandler()
    response = handler.handle(handler_input)

    assert response.output_speech.ssml == '<speak>Hi there</speak>'
    assert handler_input.attributes_manager.session_attributes['chat_history'][-1] == ('hello', 'Hi there')
    assert mock_post.called
