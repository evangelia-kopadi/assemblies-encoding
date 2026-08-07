import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.visualization.dag_plotting import visualize_three_dags


def test_visualize_three_dags_marks_spurious_edges_in_legend(tmp_path):
    output_path = tmp_path / "three_dags.png"

    fig = visualize_three_dags(
        ground_truth_edges=[("A", "B")],
        neuron_edges=[("A", "B")],
        assembly_edges=[("A", "B"), ("B", "A")],
        var_names=["A", "B"],
        save_path=output_path,
    )

    try:
        assert output_path.exists()
        assert len(fig.axes) == 3
        assert fig.axes[2].get_title() == "Assembly DAG\n(2 edges)"
        legend = fig.axes[2].get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "Correct Edge",
            "Spurious Edge",
        ]
        assert fig.axes[1].get_legend() is None
    finally:
        plt.close(fig)
