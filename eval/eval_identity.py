"""Task 3 evaluation. TODO(T3).

Suggested metrics:
  - identity switches per crossing (recorded crossings + MockPoseSource(scenario="cross"))
  - recovery: correct re-identification rate after leaving/re-entering the frame
  - spatial accuracy: estimated vs. tape-measured distance to the camera
  - time per PlayerTracker.update call

Quick start with the mock data (no camera needed):
    from core.mock_source import MockPoseSource
    mock = MockPoseSource(scenario="cross")
    poses, faces = mock.generate(t, (720, 1280, 3))   # ground truth: list index = true identity
"""


def main() -> None:
    raise SystemExit("Task 3 evaluation not implemented yet")


if __name__ == "__main__":
    main()
