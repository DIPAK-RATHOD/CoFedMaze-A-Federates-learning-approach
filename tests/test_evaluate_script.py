import json

from env.core.actions import NUM_ACTIONS
from env.core.observations import NUM_CHANNELS
from marl.models.vdn import VDNModel
from scripts.evaluate import run
from utils.checkpoint import save_checkpoint


def test_evaluate_script_writes_metrics_report_for_explicit_seed(tmp_path, capsys):
    checkpoint_dir = tmp_path / "checkpoints"
    output_path = tmp_path / "metrics.json"
    model = VDNModel(NUM_CHANNELS, window_size=5, num_actions=NUM_ACTIONS)
    save_checkpoint(checkpoint_dir, model, episode_count=12)

    result = run([
        "--node-config", "data/node1/config.yaml",
        "--checkpoint-dir", str(checkpoint_dir),
        "--seeds", "1",
        "--max-episode-steps", "2",
        "--output", str(output_path),
    ])

    assert result == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["node_id"] == "N1"
    assert report["checkpoint_episode"] == 12
    assert report["seeds"] == [1]
    assert report["metrics"]["num_episodes"] == 1
    assert '"checkpoint_slot": "current"' in capsys.readouterr().out
