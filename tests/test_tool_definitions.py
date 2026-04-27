"""Tests for tool schema registry (tools/definitions.py)."""

import pytest

from tools.definitions import TOOL_REGISTRY, get_tool_schemas, get_tool_names


def test_tool_registry_has_expected_tools():
    expected = {"check_doctor_slots", "list_doctors", "book_appointment", "escalate_to_human"}
    assert expected == set(TOOL_REGISTRY.keys())


def test_each_tool_is_function_type():
    for name, schema in TOOL_REGISTRY.items():
        assert schema["type"] == "function", f"{name} should have type 'function'"
        assert "function" in schema
        func = schema["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func


def test_get_tool_schemas_returns_list_of_dicts():
    schemas = get_tool_schemas()
    assert isinstance(schemas, list)
    assert all(isinstance(s, dict) for s in schemas)


def test_get_tool_names_returns_strings():
    names = get_tool_names()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


def test_tool_schemas_and_names_are_consistent():
    schemas = get_tool_schemas()
    names = get_tool_names()
    schema_names = [s["function"]["name"] for s in schemas]
    assert schema_names == names


def test_book_appointment_required_fields():
    schema = TOOL_REGISTRY["book_appointment"]["function"]
    required = schema["parameters"].get("required", [])
    assert "patient_name" in required
    assert "concern" in required


def test_escalate_to_human_required_reason():
    schema = TOOL_REGISTRY["escalate_to_human"]["function"]
    required = schema["parameters"].get("required", [])
    assert "reason" in required


def test_check_doctor_slots_required_doctor_name():
    schema = TOOL_REGISTRY["check_doctor_slots"]["function"]
    required = schema["parameters"].get("required", [])
    assert "doctor_name" in required


def test_list_doctors_no_required_fields():
    schema = TOOL_REGISTRY["list_doctors"]["function"]
    required = schema["parameters"].get("required", [])
    assert required == []
