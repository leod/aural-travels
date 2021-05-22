# Genre Classifier

Here, we train an image classification model to predict the gerne from track covers or album covers.

```
PYTHONPATH=. tools/train_genre_classifier.py \
    --soundcloud_data_dir <scdata-dir> \
    --dataset soundcloud \
    --num_epochs 15 \
    --batch_size 32 \
    &> logs/soundcloud/train_genre_classifier.log
```

> TODO 