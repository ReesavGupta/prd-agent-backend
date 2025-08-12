import pytest


@pytest.mark.asyncio
async def test_route_entry_sets_generate_when_prd_empty():
    from app.agent.state import AgentState
    # Lazy import nodes without initializing providers: patch jose import
    import sys, types
    # Stub security deps used at import time
    if 'jose' not in sys.modules:
        jwt_stub = types.SimpleNamespace()
        sys.modules['jose'] = types.SimpleNamespace(JWTError=Exception, jwt=jwt_stub)
    # Create a fake passlib.context module with CryptContext attr
    if 'passlib' not in sys.modules:
        pass
    if 'passlib.context' not in sys.modules:
        mod = types.ModuleType('passlib.context')
        setattr(mod, 'CryptContext', lambda **k: None)
        sys.modules['passlib.context'] = mod
    from app.agent.nodes import route_entry

    st = AgentState(project_id="p", chat_id="c", user_id="u", base_prd_markdown="", prd_markdown="")
    out = await route_entry(st)
    assert out.telemetry.get("entry_route") == "generate_prd"


@pytest.mark.asyncio
async def test_route_entry_classifies_update_instruction(monkeypatch):
    from app.agent.state import AgentState
    import sys, types
    if 'jose' not in sys.modules:
        jwt_stub = types.SimpleNamespace()
        sys.modules['jose'] = types.SimpleNamespace(JWTError=Exception, jwt=jwt_stub)
    if 'passlib.context' not in sys.modules:
        mod = types.ModuleType('passlib.context')
        setattr(mod, 'CryptContext', lambda **k: None)
        sys.modules['passlib.context'] = mod
    # Stub `motor.motor_asyncio` to avoid DB import
    if 'motor.motor_asyncio' not in sys.modules:
        mod = types.ModuleType('motor.motor_asyncio')
        setattr(mod, 'AsyncIOMotorDatabase', object)
        sys.modules['motor.motor_asyncio'] = mod
    from app.agent.nodes import route_entry
    import app.agent.nodes as nodes_mod

    async def fake_classify(user_id, user_text, section=None, question=None, idea=None, recent_qa=None):
        return ("update_section_content", {"section_ref": "11", "instructions": "add targets"})

    monkeypatch.setattr(nodes_mod, "classify_intent", fake_classify)

    st = AgentState(
        project_id="p",
        chat_id="c",
        user_id="u",
        prd_markdown="# Product Requirement Document: X\n\n### 11. Success Metrics / KPIs\n\n",
        last_messages=[{"role": "user", "content": "Update metrics with targets"}],
    )
    out = await route_entry(st)
    assert out.telemetry.get("entry_route") == "apply_user_instruction"


@pytest.mark.asyncio
async def test_route_entry_classifies_followup_question(monkeypatch):
    from app.agent.state import AgentState
    import sys, types
    if 'jose' not in sys.modules:
        jwt_stub = types.SimpleNamespace()
        sys.modules['jose'] = types.SimpleNamespace(JWTError=Exception, jwt=jwt_stub)
    if 'passlib.context' not in sys.modules:
        mod = types.ModuleType('passlib.context')
        setattr(mod, 'CryptContext', lambda **k: None)
        sys.modules['passlib.context'] = mod
    if 'motor.motor_asyncio' not in sys.modules:
        mod = types.ModuleType('motor.motor_asyncio')
        setattr(mod, 'AsyncIOMotorDatabase', object)
        sys.modules['motor.motor_asyncio'] = mod
    from app.agent.nodes import route_entry
    import app.agent.nodes as nodes_mod

    async def fake_classify(user_id, user_text, section=None, question=None, idea=None, recent_qa=None):
        return ("followup_question", {"question": "What KPIs?"})

    monkeypatch.setattr(nodes_mod, "classify_intent", fake_classify)

    st = AgentState(
        project_id="p",
        chat_id="c",
        user_id="u",
        prd_markdown="# Product Requirement Document: X\n\n### 11. Success Metrics / KPIs\n\n",
        last_messages=[{"role": "user", "content": "What KPIs are we using?"}],
    )
    out = await route_entry(st)
    assert out.telemetry.get("entry_route") == "chat_qa_once"


