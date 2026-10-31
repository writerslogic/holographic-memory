import unittest
from rename_warm import transform

class RenameCompatibilityTests(unittest.TestCase):
    def test_wire_bytes_and_claim_are_immutable(self):
        original = '''// An unmatched apostrophe shouldn't consume byte literals.
const MAGIC: &[u8; 8] = b"HMSCKPE\\x01";
const HASH = b"holographic-memory shard key 1234567";
let claim = json!({"hms_namespace": hms_namespace});
let lock = ".hms.lock";
'''
        self.assertEqual(transform(original), original)

    def test_product_identifiers_and_links_change(self):
        self.assertEqual(transform('HmsCore HolographicMemorySystem https://github.com/writerslogic/holographic-memory'),
                         'WarmCore WarmMemorySystem https://github.com/writerslogic/warm')

    def test_signed_context_remains_and_product_url_moves(self):
        self.assertEqual(transform('"https://docs.holographic-memory.com/ns/hms/v1.jsonld"'),
                         '"https://docs.holographic-memory.com/ns/hms/v1.jsonld"')
        self.assertEqual(transform('"https://holographic-memory.com"'), '"https://writerslogic.com/warm"')

if __name__ == '__main__': unittest.main()
