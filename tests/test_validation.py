"""Tests for the validation module and friendly error messages."""

import pytest

from understudy import Scene, SceneValidationError


class TestSceneValidationErrors:
    def test_missing_required_field_starting_prompt(self, tmp_path):
        scene_file = tmp_path / "test.yaml"
        scene_file.write_text("""\
id: test
conversation_plan: Do something
persona: cooperative
""")
        with pytest.raises(SceneValidationError) as exc_info:
            Scene.from_file(scene_file)

        assert "starting_prompt" in str(exc_info.value)
        assert "Missing required field" in str(exc_info.value)

    def test_missing_required_field_conversation_plan(self, tmp_path):
        scene_file = tmp_path / "test.yaml"
        scene_file.write_text("""\
id: test
starting_prompt: Hello
persona: cooperative
""")
        with pytest.raises(SceneValidationError) as exc_info:
            Scene.from_file(scene_file)

        assert "conversation_plan" in str(exc_info.value)

    def test_missing_required_field_persona(self, tmp_path):
        scene_file = tmp_path / "test.yaml"
        scene_file.write_text("""\
id: test
starting_prompt: Hello
conversation_plan: Do something
""")
        with pytest.raises(SceneValidationError) as exc_info:
            Scene.from_file(scene_file)

        assert "persona" in str(exc_info.value)

    def test_error_includes_file_path(self, tmp_path):
        scene_file = tmp_path / "my_scene.yaml"
        scene_file.write_text("""\
id: test
""")
        with pytest.raises(SceneValidationError) as exc_info:
            Scene.from_file(scene_file)

        assert "my_scene.yaml" in str(exc_info.value)

    def test_error_includes_example(self, tmp_path):
        scene_file = tmp_path / "test.yaml"
        scene_file.write_text("""\
id: test
conversation_plan: Do something
persona: cooperative
""")
        with pytest.raises(SceneValidationError) as exc_info:
            Scene.from_file(scene_file)

        assert "Example:" in str(exc_info.value)

    def test_empty_file_error(self, tmp_path):
        scene_file = tmp_path / "empty.yaml"
        scene_file.write_text("")

        with pytest.raises(SceneValidationError) as exc_info:
            Scene.from_file(scene_file)

        assert "empty" in str(exc_info.value).lower()

    def test_invalid_yaml_syntax(self, tmp_path):
        scene_file = tmp_path / "bad.yaml"
        scene_file.write_text("""\
id: test
  bad_indent: this is invalid
""")
        with pytest.raises(SceneValidationError) as exc_info:
            Scene.from_file(scene_file)

        assert "Invalid YAML syntax" in str(exc_info.value) or "YAML" in str(exc_info.value)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Scene.from_file(tmp_path / "nonexistent.yaml")


class TestCommonMistakeWarnings:
    def test_warns_about_prompt_vs_starting_prompt(self, tmp_path, capsys):
        scene_file = tmp_path / "test.yaml"
        scene_file.write_text("""\
id: test
prompt: Hello
conversation_plan: Do something
persona: cooperative
""")
        with pytest.raises(SceneValidationError):
            Scene.from_file(scene_file)

        captured = capsys.readouterr()
        assert "starting_prompt" in captured.err or "prompt" in captured.err

    def test_warns_about_unknown_persona_preset(self, tmp_path, capsys):
        scene_file = tmp_path / "test.yaml"
        scene_file.write_text("""\
id: test
starting_prompt: Hello
conversation_plan: Do something
persona: unknown_preset
""")
        with pytest.raises(Exception):
            Scene.from_file(scene_file)

        captured = capsys.readouterr()
        assert "unknown_preset" in captured.err or "cooperative" in captured.err


class TestValidSceneLoading:
    def test_valid_scene_loads_without_error(self, tmp_path):
        scene_file = tmp_path / "valid.yaml"
        scene_file.write_text("""\
id: test_scene
description: A valid test scene
starting_prompt: Hello, I need help
conversation_plan: Ask for help and complete the task
persona: cooperative
max_turns: 10
expectations:
  required_tools:
    - lookup_order
""")
        scene = Scene.from_file(scene_file)
        assert scene.id == "test_scene"
        assert scene.starting_prompt == "Hello, I need help"
        assert "lookup_order" in scene.expectations.required_tools

    def test_custom_persona_loads(self, tmp_path):
        scene_file = tmp_path / "custom_persona.yaml"
        scene_file.write_text("""\
id: test
starting_prompt: Hello
conversation_plan: Test
persona:
  description: A custom persona
  behaviors:
    - Always agrees
    - Never complains
""")
        scene = Scene.from_file(scene_file)
        assert scene.persona.description == "A custom persona"
        assert len(scene.persona.behaviors) == 2
