from pipeline.scope_guard import (
    caller_insists_on_human,
    decline_message,
    detect_out_of_scope,
    should_escalate_oos,
    transfer_message,
)


def test_detect_billing():
    assert detect_out_of_scope("I want a refund on my bill") == "billing"


def test_detect_lab_hindi():
    assert detect_out_of_scope("mera lab report kab aayega") == "lab"


def test_in_scope_appointment():
    assert detect_out_of_scope("book appointment with Dr. Sharma kal") is None


def test_in_scope_consultation_fee():
    assert detect_out_of_scope("consultation fee kitna hai Dr. Sharma ke liye") is None


def test_insist_human():
    assert caller_insists_on_human("please connect me to a human agent") is True


def test_no_insist():
    assert caller_insists_on_human("kal appointment chahiye") is False


def test_should_escalate_repeat_category():
    assert should_escalate_oos(
        category="billing",
        strikes=1,
        last_category="billing",
        insists=False,
    ) is True


def test_should_not_escalate_first_strike():
    assert should_escalate_oos(
        category="billing",
        strikes=0,
        last_category=None,
        insists=False,
    ) is False


def test_decline_message_hindi():
    msg = decline_message("billing", "hindi")
    assert "अपॉइंटमेंट" in msg


def test_transfer_message_english():
    assert "team" in transfer_message("english").lower()
