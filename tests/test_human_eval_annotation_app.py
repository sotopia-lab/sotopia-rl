import tempfile
import unittest
from pathlib import Path

from human_eval.episode_parser import episode_files, parse_episode_file
from human_eval.storage import build_annotation_document, load_annotation, save_annotation

REPO_ROOT = Path(__file__).resolve().parents[1]
HUMAN_EVAL_DIR = REPO_ROOT / 'human_eval'


class HumanEvalAnnotationAppTests(unittest.TestCase):
    def test_all_markdown_episodes_parse(self) -> None:
        files = episode_files(HUMAN_EVAL_DIR)
        self.assertTrue(files)

        for path in files:
            episode = parse_episode_file(path)
            self.assertEqual(episode['episode_id'], path.stem)
            self.assertEqual(len(episode['agents']), 2)
            self.assertEqual(episode['turn_count'], len(episode['turns']))
            self.assertTrue(episode['turn_keys'][0].startswith('agent_'))

    def test_action_and_event_lines_are_classified(self) -> None:
        action_episode = parse_episode_file(
            HUMAN_EVAL_DIR / 'split_1' / '01JV3KB8CZTFYN1YF70NCMCWDP.md'
        )
        self.assertTrue(any(turn['content_type'] == 'action' for turn in action_episode['turns']))

        event_episode = parse_episode_file(
            HUMAN_EVAL_DIR / 'split_1' / '01JV059EEHEVQXEM124R5ZNMFH.md'
        )
        self.assertTrue(any(turn['content_type'] == 'event' for turn in event_episode['turns']))

    def test_build_annotation_document_tracks_missing_turns(self) -> None:
        episode = parse_episode_file(
            HUMAN_EVAL_DIR / 'split_2' / '01JV3WVCG3TJ5MCBHE7A3PNWGG.md'
        )
        first_turn = episode['turns'][0]['turn_key']

        document = build_annotation_document(
            episode,
            {
                'annotator': 'jq',
                'final_goal_achieving_score': '',
                'reward_attribution': {first_turn: 7},
            },
        )

        self.assertEqual(document['status'], 'draft')
        self.assertEqual(document['reward_attribution'], {first_turn: 7})
        self.assertNotIn(first_turn, document['missing_reward_attribution_keys'])
        self.assertEqual(
            len(document['missing_reward_attribution_keys']),
            episode['turn_count'] - 1,
        )

    def test_save_annotation_round_trip(self) -> None:
        episode = parse_episode_file(
            HUMAN_EVAL_DIR / 'split_2' / '01JV3WVCG3TJ5MCBHE7A3PNWGG.md'
        )
        payload = {
            'annotator': 'tester',
            'notes': 'round-trip check',
            'final_goal_achieving_score': 6,
            'reward_attribution': {turn['turn_key']: 0 for turn in episode['turns']},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            saved = save_annotation(episode, payload, temp_path)
            loaded = load_annotation(episode['split'], episode['episode_id'], temp_path)

        self.assertEqual(saved['status'], 'completed')
        self.assertIsNotNone(saved['completed_at'])
        self.assertEqual(saved['reward_attribution'], payload['reward_attribution'])
        self.assertEqual(loaded, saved)


if __name__ == '__main__':
    unittest.main()
