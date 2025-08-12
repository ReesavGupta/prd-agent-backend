import asyncio
import json
import pytest

from app.agent.intent import classify_intent, verify_is_answer


@pytest.mark.asyncio
async def test_classify_intent_update_section_content(monkeypatch):
    async def fake_generate_response(user_id, messages, temperature=None, max_tokens=None, use_cache=None, **kwargs):
        class R:
            content = json.dumps({
                "intent": "update_section_content",
                "args": {"section_ref": "11", "instructions": "add activation and retention targets"}
            })
        return R()

    # Patch the lazy getter to return an object with generate_response
    import app.agent.intent as intent_mod
    class FakeAI: pass
    fake = FakeAI()
    fake.generate_response = fake_generate_response
    monkeypatch.setattr(intent_mod, "_get_ai", lambda: fake)

    intent, args = await classify_intent(
        user_id="u1",
        user_text="Update Success Metrics with activation and retention targets",
        section="11. Success Metrics / KPIs",
        question="Which KPIs will we use?",
    )
    assert intent == "update_section_content"
    assert args.get("section_ref") == "11"
    assert "activation" in args.get("instructions", "")


@pytest.mark.asyncio
async def test_classify_intent_followup_question(monkeypatch):
    async def fake_generate_response(user_id, messages, temperature=None, max_tokens=None, use_cache=None, **kwargs):
        class R:
            content = json.dumps({
                "intent": "followup_question",
                "args": {"question": "What KPIs are we using?", "target_section_guess": "11"}
            })
        return R()

    import app.agent.intent as intent_mod
    class FakeAI: pass
    fake = FakeAI()
    fake.generate_response = fake_generate_response
    monkeypatch.setattr(intent_mod, "_get_ai", lambda: fake)

    intent, args = await classify_intent(
        user_id="u1",
        user_text="What KPIs are we using?",
        section="11. Success Metrics / KPIs",
        question="",
    )
    assert intent == "followup_question"
    assert args.get("question")


@pytest.mark.asyncio
async def test_verify_is_answer_true(monkeypatch):
    async def fake_generate_response(user_id, messages, temperature=None, max_tokens=None, use_cache=None, **kwargs):
        class R:
            content = json.dumps({"is_answer": True})
        return R()

    import app.agent.intent as intent_mod
    class FakeAI: pass
    fake = FakeAI()
    fake.generate_response = fake_generate_response
    monkeypatch.setattr(intent_mod, "_get_ai", lambda: fake)

    ok = await verify_is_answer("u1", "Primary KPI will be WAU and retention at 8 weeks.", "11. Success Metrics / KPIs", "Which KPIs will we use?")
    assert ok is True


