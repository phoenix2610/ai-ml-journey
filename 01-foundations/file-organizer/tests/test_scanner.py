import pytest

from organizer.scanner import file_hash, find_duplicates, human_size, scan


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.txt").write_text("hello")  # duplicate of a
    (tmp_path / "c.txt").write_text("different")
    (tmp_path / ".hidden").write_text("x")
    (tmp_path / "empty1").write_text("")
    (tmp_path / "empty2").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.txt").write_text("nested")
    skipped = tmp_path / "__pycache__"
    skipped.mkdir()
    (skipped / "junk.pyc").write_text("junk")
    return tmp_path


def names(infos):
    return sorted(i.name for i in infos)


def test_scan_is_shallow_by_default(tree):
    assert "d.txt" not in names(scan(tree))


def test_scan_recursive(tree):
    assert "d.txt" in names(scan(tree, recursive=True))


def test_hidden_files_skipped_by_default(tree):
    assert ".hidden" not in names(scan(tree))


def test_hidden_files_can_be_included(tree):
    assert ".hidden" in names(scan(tree, include_hidden=True))


def test_skip_dirs_are_ignored(tree):
    assert "junk.pyc" not in names(scan(tree, recursive=True))


def test_directories_are_not_yielded(tree):
    assert "sub" not in names(scan(tree))


def test_symlinks_skipped_by_default(tree):
    (tree / "link.txt").symlink_to(tree / "a.txt")
    assert "link.txt" not in names(scan(tree))


def test_scan_on_a_file_raises(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        list(scan(f))


def test_file_hash_is_content_based(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("same")
    b.write_text("same")
    assert file_hash(a) == file_hash(b)


def test_file_hash_differs_on_different_content(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_text("one")
    b.write_text("two")
    assert file_hash(a) != file_hash(b)


def test_hash_is_chunk_size_independent(tmp_path):
    f = tmp_path / "big"
    f.write_bytes(b"x" * 5000)
    assert file_hash(f, chunk=16) == file_hash(f, chunk=4096)


def test_find_duplicates_groups_identical_content(tree):
    groups = find_duplicates(scan(tree))
    assert len(groups) == 1
    assert names(next(iter(groups.values()))) == ["a.txt", "b.txt"]


def test_empty_files_are_not_reported_as_duplicates(tree):
    # Every empty file matches every other; reporting them is just noise.
    all_dupes = [i.name for g in find_duplicates(scan(tree)).values() for i in g]
    assert "empty1" not in all_dupes


def test_unique_files_produce_no_groups(tmp_path):
    (tmp_path / "a").write_text("one")
    (tmp_path / "b").write_text("two")
    assert find_duplicates(scan(tmp_path)) == {}


@pytest.mark.parametrize(
    "size,text", [(0, "0 B"), (512, "512 B"), (2048, "2.0 KB"), (5 * 1024**2, "5.0 MB")]
)
def test_human_size(size, text):
    assert human_size(size) == text
