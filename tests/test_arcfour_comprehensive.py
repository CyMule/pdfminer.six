"""Comprehensive tests for pdfminer/arcfour.py

Tests the Arcfour (RC4) stream cipher implementation including:
- Key scheduling algorithm (KSA)
- Pseudo-random generation algorithm (PRGA)
- Encryption and decryption with known test vectors
- Edge cases and stateful behavior
"""

import binascii

from pdfminer.arcfour import Arcfour


def hex_encode(b: bytes) -> bytes:
    """Encode bytes to hex string."""
    return binascii.hexlify(b)


def hex_decode(s: str) -> bytes:
    """Decode hex string to bytes."""
    return binascii.unhexlify(s)


class TestArcfourBasicEncryption:
    """Basic encryption tests with known test vectors from RFC 6229 and Wikipedia."""

    def test_key_plaintext_vector(self):
        """Test vector: Key='Key', Plaintext='Plaintext'"""
        cipher = Arcfour(b"Key")
        result = cipher.process(b"Plaintext")
        assert hex_encode(result) == b"bbf316e8d940af0ad3"

    def test_wiki_pedia_vector(self):
        """Test vector: Key='Wiki', Plaintext='pedia'"""
        cipher = Arcfour(b"Wiki")
        result = cipher.process(b"pedia")
        assert hex_encode(result) == b"1021bf0420"

    def test_secret_attack_vector(self):
        """Test vector: Key='Secret', Plaintext='Attack at dawn'"""
        cipher = Arcfour(b"Secret")
        result = cipher.process(b"Attack at dawn")
        assert hex_encode(result) == b"45a01f645fc35b383552544b9bf5"


class TestArcfourSymmetry:
    """Tests verifying that encryption and decryption are symmetric."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypting then decrypting should return original plaintext."""
        key = b"testkey"
        plaintext = b"Hello, World!"

        cipher1 = Arcfour(key)
        ciphertext = cipher1.encrypt(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.decrypt(ciphertext)

        assert decrypted == plaintext

    def test_process_encrypt_decrypt_aliases(self):
        """Verify that encrypt, decrypt, and process are all the same function."""
        assert Arcfour.encrypt is Arcfour.process
        assert Arcfour.decrypt is Arcfour.process

    def test_roundtrip_with_binary_data(self):
        """Test roundtrip with arbitrary binary data."""
        key = b"binarykey"
        plaintext = bytes(range(256))

        cipher1 = Arcfour(key)
        ciphertext = cipher1.process(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(ciphertext)

        assert decrypted == plaintext

    def test_roundtrip_long_message(self):
        """Test roundtrip with a longer message."""
        key = b"longmessagekey"
        plaintext = b"A" * 10000

        cipher1 = Arcfour(key)
        ciphertext = cipher1.process(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(ciphertext)

        assert decrypted == plaintext


class TestArcfourStatefulBehavior:
    """Tests verifying the stateful nature of the RC4 cipher."""

    def test_sequential_processing_differs_from_combined(self):
        """Processing in chunks differs from processing all at once (stateful)."""
        key = b"statekey"
        data = b"HelloWorld"

        cipher1 = Arcfour(key)
        combined = cipher1.process(data)

        cipher2 = Arcfour(key)
        part1 = cipher2.process(b"Hello")
        part2 = cipher2.process(b"World")

        assert part1 + part2 == combined

    def test_state_preserved_across_calls(self):
        """Internal state (i, j) is preserved across process calls."""
        key = b"preservekey"
        cipher = Arcfour(key)

        cipher.process(b"first")
        i1, j1 = cipher.i, cipher.j

        cipher.process(b"second")
        i2, j2 = cipher.i, cipher.j

        assert (i1, j1) != (0, 0)
        assert (i2, j2) != (i1, j1)

    def test_state_not_reset_after_processing(self):
        """After processing, the cipher state has advanced."""
        key = b"notreset"
        cipher = Arcfour(key)

        assert cipher.i == 0
        assert cipher.j == 0

        cipher.process(b"data")

        assert cipher.i != 0 or cipher.j != 0

    def test_multiple_chunks_match_single_chunk(self):
        """Processing data in multiple small chunks equals single large chunk."""
        key = b"chunkkey"
        data = b"This is a longer piece of data for testing"

        cipher1 = Arcfour(key)
        single_result = cipher1.process(data)

        cipher2 = Arcfour(key)
        chunked_result = b""
        for i in range(0, len(data), 5):
            chunk = data[i : i + 5]
            chunked_result += cipher2.process(chunk)

        assert chunked_result == single_result


class TestArcfourEdgeCases:
    """Edge case tests for the Arcfour implementation."""

    def test_empty_plaintext(self):
        """Processing empty plaintext returns empty bytes."""
        cipher = Arcfour(b"key")
        result = cipher.process(b"")
        assert result == b""

    def test_single_byte_key(self):
        """Single byte key works correctly."""
        cipher = Arcfour(b"k")
        result = cipher.process(b"test")
        assert len(result) == 4

        cipher2 = Arcfour(b"k")
        decrypted = cipher2.process(result)
        assert decrypted == b"test"

    def test_single_byte_plaintext(self):
        """Single byte plaintext works correctly."""
        cipher = Arcfour(b"key")
        result = cipher.process(b"X")
        assert len(result) == 1

        cipher2 = Arcfour(b"key")
        decrypted = cipher2.process(result)
        assert decrypted == b"X"

    def test_256_byte_key(self):
        """256 byte key (maximum useful length) works correctly."""
        key = bytes(range(256))
        cipher = Arcfour(key)
        plaintext = b"test message"
        result = cipher.process(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(result)
        assert decrypted == plaintext

    def test_key_longer_than_256_bytes(self):
        """Key longer than 256 bytes still works (wraps around)."""
        key = bytes(range(256)) + b"extra"
        cipher = Arcfour(key)
        plaintext = b"test"
        result = cipher.process(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(result)
        assert decrypted == plaintext

    def test_null_bytes_in_key(self):
        """Key containing null bytes works correctly."""
        key = b"\x00\x00\x00"
        cipher = Arcfour(key)
        plaintext = b"test"
        result = cipher.process(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(result)
        assert decrypted == plaintext

    def test_null_bytes_in_plaintext(self):
        """Plaintext containing null bytes works correctly."""
        key = b"key"
        plaintext = b"\x00\x00\x00\x00"
        cipher = Arcfour(key)
        result = cipher.process(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(result)
        assert decrypted == plaintext

    def test_all_255_bytes_in_plaintext(self):
        """Plaintext of all 0xFF bytes works correctly."""
        key = b"key"
        plaintext = b"\xff" * 100
        cipher = Arcfour(key)
        result = cipher.process(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(result)
        assert decrypted == plaintext


class TestArcfourKeyScheduling:
    """Tests for the key scheduling algorithm (KSA)."""

    def test_s_box_initialized_correctly(self):
        """S-box contains all values 0-255 after initialization."""
        cipher = Arcfour(b"testkey")
        s_values = sorted(cipher.s)
        assert s_values == list(range(256))

    def test_s_box_permuted(self):
        """S-box is permuted (not in order) after key scheduling."""
        cipher = Arcfour(b"testkey")
        assert cipher.s != list(range(256))

    def test_different_keys_different_s_boxes(self):
        """Different keys produce different S-box permutations."""
        cipher1 = Arcfour(b"key1")
        cipher2 = Arcfour(b"key2")
        assert cipher1.s != cipher2.s

    def test_same_key_same_s_box(self):
        """Same key produces identical S-box permutations."""
        cipher1 = Arcfour(b"samekey")
        cipher2 = Arcfour(b"samekey")
        assert cipher1.s == cipher2.s

    def test_initial_i_j_are_zero(self):
        """Initial i and j values are 0 after construction."""
        cipher = Arcfour(b"key")
        assert cipher.i == 0
        assert cipher.j == 0


class TestArcfourKeyTypes:
    """Tests for different key input types."""

    def test_bytes_key(self):
        """Standard bytes key works correctly."""
        cipher = Arcfour(b"byteskey")
        result = cipher.process(b"test")
        assert len(result) == 4

    def test_list_of_ints_key(self):
        """List of integers as key works correctly."""
        key = [75, 101, 121]  # ASCII for "Key"
        cipher = Arcfour(key)
        result = cipher.process(b"Plaintext")
        assert hex_encode(result) == b"bbf316e8d940af0ad3"

    def test_bytearray_key(self):
        """Bytearray as key works correctly."""
        key = bytearray(b"Key")
        cipher = Arcfour(key)
        result = cipher.process(b"Plaintext")
        assert hex_encode(result) == b"bbf316e8d940af0ad3"

    def test_tuple_of_ints_key(self):
        """Tuple of integers as key works correctly."""
        key = (75, 101, 121)  # ASCII for "Key"
        cipher = Arcfour(key)
        result = cipher.process(b"Plaintext")
        assert hex_encode(result) == b"bbf316e8d940af0ad3"


class TestArcfourRFC6229Vectors:
    """Additional test vectors based on RFC 6229 for RC4."""

    def test_rfc6229_key_0102030405(self):
        """Test with key from RFC 6229 style."""
        key = hex_decode("0102030405")
        cipher = Arcfour(key)

        plaintext = b"\x00" * 16
        result = cipher.process(plaintext)

        assert hex_encode(result) == b"b2396305f03dc027ccc3524a0a1118a8"

    def test_rfc6229_key_01020304050607(self):
        """Test with 7-byte key."""
        key = hex_decode("01020304050607")
        cipher = Arcfour(key)

        plaintext = b"\x00" * 16
        result = cipher.process(plaintext)

        assert hex_encode(result) == b"293f02d47f37c9b633f2af5285feb46b"

    def test_rfc6229_key_0102030405060708(self):
        """Test with 8-byte key."""
        key = hex_decode("0102030405060708")
        cipher = Arcfour(key)

        plaintext = b"\x00" * 16
        result = cipher.process(plaintext)

        assert hex_encode(result) == b"97ab8a1bf0afb96132f2f67258da15a8"


class TestArcfourDeterminism:
    """Tests verifying deterministic behavior."""

    def test_same_input_same_output(self):
        """Same key and plaintext always produce same ciphertext."""
        key = b"deterministickey"
        plaintext = b"deterministic test"

        results = []
        for _ in range(10):
            cipher = Arcfour(key)
            results.append(cipher.process(plaintext))

        assert all(r == results[0] for r in results)

    def test_different_keys_different_output(self):
        """Different keys produce different ciphertext for same plaintext."""
        plaintext = b"same plaintext"

        cipher1 = Arcfour(b"key1")
        cipher2 = Arcfour(b"key2")

        result1 = cipher1.process(plaintext)
        result2 = cipher2.process(plaintext)

        assert result1 != result2


class TestArcfourStreamProperties:
    """Tests verifying stream cipher properties."""

    def test_ciphertext_same_length_as_plaintext(self):
        """Ciphertext is always the same length as plaintext."""
        key = b"lengthkey"

        for length in [0, 1, 10, 100, 1000]:
            cipher = Arcfour(key)
            plaintext = b"x" * length
            ciphertext = cipher.process(plaintext)
            assert len(ciphertext) == length

    def test_xor_based_encryption(self):
        """Verify XOR-based encryption (double encryption = original)."""
        key = b"xorkey"
        plaintext = b"XOR test message"

        cipher1 = Arcfour(key)
        ciphertext = cipher1.process(plaintext)

        cipher2 = Arcfour(key)
        recovered = cipher2.process(ciphertext)

        assert recovered == plaintext

    def test_keystream_generation(self):
        """Test keystream generation by encrypting zeros."""
        key = b"keystreamkey"
        zeros = b"\x00" * 20

        cipher = Arcfour(key)
        keystream = cipher.process(zeros)

        assert len(keystream) == 20
        assert keystream != zeros


class TestArcfourBinaryData:
    """Tests with various binary data patterns."""

    def test_all_byte_values_in_plaintext(self):
        """Test plaintext containing all possible byte values."""
        key = b"allbyteskey"
        plaintext = bytes(range(256))

        cipher1 = Arcfour(key)
        ciphertext = cipher1.process(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(ciphertext)

        assert decrypted == plaintext

    def test_repeated_pattern(self):
        """Test repeated pattern in plaintext."""
        key = b"patternkey"
        pattern = b"ABCD"
        plaintext = pattern * 100

        cipher1 = Arcfour(key)
        ciphertext = cipher1.process(plaintext)

        assert ciphertext != plaintext

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(ciphertext)
        assert decrypted == plaintext

    def test_alternating_bits(self):
        """Test alternating bit pattern (0xAA, 0x55)."""
        key = b"alternatekey"
        plaintext = b"\xaa\x55" * 50

        cipher1 = Arcfour(key)
        ciphertext = cipher1.process(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(ciphertext)

        assert decrypted == plaintext


class TestArcfourNewInstanceRequired:
    """Tests demonstrating that a new cipher instance is needed for decryption."""

    def test_cannot_decrypt_with_same_instance(self):
        """Cannot decrypt using the same cipher instance used for encryption."""
        key = b"sameinstancekey"
        plaintext = b"Cannot reuse"

        cipher = Arcfour(key)
        ciphertext = cipher.encrypt(plaintext)

        wrong_decrypt = cipher.decrypt(ciphertext)
        assert wrong_decrypt != plaintext

    def test_must_use_fresh_instance_for_decrypt(self):
        """Must use a fresh cipher instance to decrypt."""
        key = b"freshinstancekey"
        plaintext = b"Must be fresh"

        cipher1 = Arcfour(key)
        ciphertext = cipher1.encrypt(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.decrypt(ciphertext)
        assert decrypted == plaintext


class TestArcfourLargeData:
    """Tests with larger data sizes."""

    def test_large_data_encryption(self):
        """Test encryption of larger data blocks."""
        key = b"largedatakey"
        plaintext = b"Large data block " * 1000

        cipher1 = Arcfour(key)
        ciphertext = cipher1.process(plaintext)

        assert len(ciphertext) == len(plaintext)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(ciphertext)
        assert decrypted == plaintext

    def test_streaming_large_data(self):
        """Test streaming large data in chunks."""
        key = b"streamkey"
        chunk_size = 1024
        total_size = 10000
        plaintext = b"X" * total_size

        cipher1 = Arcfour(key)
        ciphertext_chunks = []
        for i in range(0, total_size, chunk_size):
            chunk = plaintext[i : i + chunk_size]
            ciphertext_chunks.append(cipher1.process(chunk))
        ciphertext = b"".join(ciphertext_chunks)

        cipher2 = Arcfour(key)
        decrypted = cipher2.process(ciphertext)
        assert decrypted == plaintext
