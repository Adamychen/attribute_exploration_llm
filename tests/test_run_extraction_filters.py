"""Selection filters of experiments.run_extraction (dry runs, no API calls)."""
import asyncio

import pytest

from experiments.run_extraction import _read_configs, main_experiment


def dry(tmp_path, **kw):
    asyncio.run(main_experiment(output_path=str(tmp_path / "r.csv"), traces_dir=str(tmp_path / "t"),
                                dry_run=True, **kw))


def test_configs_restrict_the_plan(tmp_path, capsys):
    dry(tmp_path, configs=(("llama3.1:8b", "existential", "graphs", "basic"),))
    assert "Dry run: 1 experiments planned" in capsys.readouterr().out


def test_question_modes_restrict_the_plan(tmp_path, capsys):
    dry(tmp_path, models_filter=("llama3.1:8b",), contexts=("graphs",), prompt_modes=("basic",),
        question_modes=("implicational",))
    assert "Dry run: 1 experiments planned" in capsys.readouterr().out


def test_configs_outside_the_grid_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        dry(tmp_path, contexts=("integers",), configs=(("llama3.1:8b", "existential", "graphs", "basic"),))
    with pytest.raises(ValueError):
        dry(tmp_path, question_modes=("sideways",))


def test_read_configs(tmp_path):
    p = tmp_path / "c.csv"
    p.write_text("model,context_name,prompt_mode,question_mode,reason\nqwen3:32b,graphs,basic,existential,x\n")
    assert _read_configs(str(p)) == (("qwen3:32b", "existential", "graphs", "basic"),)
