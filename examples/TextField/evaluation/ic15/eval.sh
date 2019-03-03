# dir setting
buildDir='../../../../build/examples/TextField'
imageDir='../../../../data/icdar2015/test_images'
segDir='seg'
bboxDir='bbox'

# configs for network inference
deploy='../../deploy.prototxt'
model='../../../../models/ic15_iter_40000.caffemodel'
gpu='0'

# params for post-processing
lambda='0.75'

# init
if [ -d $segDir ]
then
    rm $segDir/*
else
    mkdir $segDir
fi
if [ -d $bboxDir ]
then
    rm $bboxDir/*
else
    mkdir $bboxDir
fi
rm submit.zip

# apply post-processing
$buildDir/inference.bin $deploy $model $gpu $imageDir/ 720 1280 $lambda $segDir/

# seg2bbox
python seg2bbox.py $segDir/ $bboxDir/ 200
zip -j submit.zip $bboxDir/*

# evaluation protocol
python script.py gt.zip submit.zip
