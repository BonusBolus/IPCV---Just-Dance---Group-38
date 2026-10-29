"""The measurements of tests/eval_identity.py: accuracy, switches, success and recovery time."""
import unittest

from eval_identity import IdentityEvaluator, left_right_ids
from identity_tracking.identity import Person, new_player_entry

CLEAN, EVENT = True, False


def run(evaluator, frames, start=0.0, step=0.1):
    """frames: list of (clean, ids). Returns the time after the last frame."""
    t = start
    for clean, ids in frames:
        evaluator.add(t, clean, ids if clean else None)
        t += step
    return t


class TestIdentityEvaluator(unittest.TestCase):
    def test_start_sets_the_ground_truth(self):
        evaluator = IdentityEvaluator()
        run(evaluator, [(CLEAN, (1, 2))] * 10)
        self.assertEqual(evaluator.truth, (1, 2))
        self.assertEqual(evaluator.accuracy, 100.0)
        self.assertEqual(evaluator.evaluated, 20)            # 2 people x 10 frames

    def test_good_crossing(self):
        """The players cross (swap sides) and the tracker swaps the IDs with them: success."""
        evaluator = IdentityEvaluator()
        t = run(evaluator, [(CLEAN, (1, 2))] * 5)
        evaluator.start_trial("crossing", t)
        t = run(evaluator, [(CLEAN, (1, 2))] * 3 + [(EVENT, None)] * 4 + [(CLEAN, (2, 1))] * 5, t)
        trial = evaluator.end_trial(t)
        self.assertTrue(trial.success)
        self.assertTrue(trial.recovered)
        self.assertAlmostEqual(trial.recovery_time, 0.0)
        self.assertEqual(trial.switches, 0)
        self.assertEqual(evaluator.truth, (2, 1))
        self.assertEqual(evaluator.accuracy, 100.0)

    def test_crossing_with_an_id_switch(self):
        """After the crossing the IDs stay on the same side: the IDs went to the wrong person."""
        evaluator = IdentityEvaluator()
        t = run(evaluator, [(CLEAN, (1, 2))] * 5)
        evaluator.start_trial("crossing", t)
        t = run(evaluator, [(EVENT, None)] * 4 + [(CLEAN, (1, 2))] * 5, t)
        trial = evaluator.end_trial(t)
        self.assertFalse(trial.success)
        self.assertFalse(trial.recovered)
        self.assertEqual(trial.switches, 1)
        self.assertEqual(evaluator.switches, 1)
        self.assertAlmostEqual(evaluator.accuracy, 100 * 10 / 20)   # 5 good frames, 5 wrong frames

    def test_recovery_time(self):
        """Wrong for 3 frames after the occlusion, then correct: recovered after 0.3 s."""
        evaluator = IdentityEvaluator()
        t = run(evaluator, [(CLEAN, (1, 2))] * 5)
        evaluator.start_trial("occlusion", t)
        t = run(evaluator, [(EVENT, None)] * 4 + [(CLEAN, (2, 1))] * 3 + [(CLEAN, (1, 2))] * 5, t)
        trial = evaluator.end_trial(t)
        self.assertTrue(trial.success)
        self.assertTrue(trial.recovered)
        self.assertAlmostEqual(trial.recovery_time, 0.3)
        self.assertEqual(trial.switches, 1)

    def test_re_entry_without_player_id(self):
        """The person comes back but gets no player (stranger): not recovered."""
        evaluator = IdentityEvaluator()
        t = run(evaluator, [(CLEAN, (1, 2))] * 5)
        evaluator.start_trial("re-entry", t)
        t = run(evaluator, [(EVENT, None)] * 10 + [(CLEAN, (1, 0))] * 5, t)
        trial = evaluator.end_trial(t)
        self.assertFalse(trial.success)
        self.assertFalse(trial.recovered)

    def test_s_key_in_a_trial(self):
        """Re-entry on the other side: s makes the trial expect a swap."""
        evaluator = IdentityEvaluator()
        t = run(evaluator, [(CLEAN, (1, 2))] * 5)
        evaluator.start_trial("re-entry", t)
        evaluator.swap_sides()
        t = run(evaluator, [(EVENT, None)] * 10 + [(CLEAN, (2, 1))] * 5, t)
        self.assertTrue(evaluator.end_trial(t).success)

    def test_trial_ended_during_the_event_is_not_valid(self):
        evaluator = IdentityEvaluator()
        t = run(evaluator, [(CLEAN, (1, 2))] * 5)
        evaluator.start_trial("crossing", t)
        t = run(evaluator, [(EVENT, None)] * 4, t)
        trial = evaluator.end_trial(t)
        self.assertIsNone(trial.success)
        self.assertEqual(evaluator.summary()["trials"]["crossing"]["valid_trials"], 0)

    def test_summary(self):
        evaluator = IdentityEvaluator()
        t = run(evaluator, [(CLEAN, (1, 2))] * 5)
        for frames in ([(EVENT, None)] * 4 + [(CLEAN, (2, 1))] * 5,      # good crossing
                       [(EVENT, None)] * 4 + [(CLEAN, (2, 1))] * 5):     # an ID switch (sides were already swapped)
            evaluator.start_trial("crossing", t)
            t = evaluator.end_trial(run(evaluator, frames, t)).end
        crossing = evaluator.summary()["trials"]["crossing"]
        self.assertEqual(crossing["valid_trials"], 2)
        self.assertEqual(crossing["success_rate_pct"], 50.0)
        self.assertEqual(crossing["switches_per_trial"], 0.5)


class TestLeftRightIds(unittest.TestCase):
    def players(self, boxes):
        entries = {i: new_player_entry(i) for i in (1, 2)}
        for player_id, box in boxes.items():
            entries[player_id].update(visible=True, box=box)
        return entries

    def test_ids_from_left_to_right(self):
        left, right = Person((100, 50, 200, 400), None), Person((500, 50, 200, 400), None)
        players = self.players({2: left.box, 1: right.box})
        self.assertEqual(left_right_ids([right, left], players), (True, (2, 1)))

    def test_overlapping_boxes_are_not_clean(self):
        a, b = Person((100, 50, 200, 400), None), Person((200, 50, 200, 400), None)
        self.assertEqual(left_right_ids([a, b], self.players({1: a.box, 2: b.box})), (False, None))

    def test_one_person_is_not_clean(self):
        a = Person((100, 50, 200, 400), None)
        self.assertEqual(left_right_ids([a], self.players({1: a.box})), (False, None))

    def test_person_without_player(self):
        a, b = Person((100, 50, 200, 400), None), Person((500, 50, 200, 400), None)
        self.assertEqual(left_right_ids([a, b], self.players({1: a.box})), (True, (1, 0)))


if __name__ == "__main__":
    unittest.main()
