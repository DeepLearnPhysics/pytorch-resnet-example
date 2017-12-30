# PyTorch 5-particle Classifier Example

Example of training using on 5-particle practice sample

Requires

* pytorch, of course
* ROOT6
* LArCV2
* pytorch interface, [LArCVDataset](https://github.com/DeepLearnPhysics/larcvdataset)

Also, download the training and validation sets from the [open data webpage](http://deeplearnphysics.org/DataChallenge/)

* [Training](http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/practice_train_5k.root)
* [Validation](http://www.stanford.edu/~kterao/public_data/v0.1.0/2d/classification/five_particles/practice_test_5k.root)


Note: as it stands, network learns, but overtrains.  Working on setting proper meta-parameters and/or adding data-augmentation.

Also, you might need to set the GPU device ID in the shell. For example, to set to device `1`,

    export CUDA_VISIBLE_DEVICES=1


### Sources

* `main.py` derives from the pytorch examples [repo](https://github.com/pytorch/examples/blob/master/imagenet/main.py)
* `resnet_example.py` is modified from the pytorch torchvision models resnet module



