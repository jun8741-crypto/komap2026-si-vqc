from qsi import io


def test_split_excludes_valid_alloys(data_splits):
    train, test, overlap = data_splits
    assert overlap == [7, 40, 55, 125, 153, 181, 191]
    assert train.image.nunique() == 61
    assert (train.usable == 1).sum() == 54095
    assert test.image.nunique() == 20
    assert len(test) == 22788
    assert not set(train.alloy) & set(test.alloy)


def test_sampling_reproduces_original_indices(data_splits):
    train, _, _ = data_splits
    sample = io.balanced_sample(train, 256, 0)
    assert list(sample.index[:5]) == [5993, 4994, 44637, 45223, 32223]
    assert sample.label.value_counts().to_dict() == {1: 256, 0: 256}
    assert sample.index.is_unique
    assert sample.index.equals(io.balanced_sample(train, 256, 0).index)
    assert list(io.threshold_set(train).index[:5]) == [28492, 22111, 29864, 44222, 61145]
