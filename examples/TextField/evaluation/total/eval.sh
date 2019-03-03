# dir setting
buildDir='../../../../build/examples/TextField'
imageDir='../../../../data/totaltext/Images/Test'
gtDir='../../../../data/totaltext/Groundtruth/Polygon/Test'
segDir='seg'
bboxDir='bbox'

# configs for network inference
deploy='../../deploy.prototxt'
model='../../../../models/total_iter_40000.caffemodel'
gpu='0'

# params for post-processing
lambda='0.50'

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

# apply post-processing
$buildDir/inference.bin $deploy $model $gpu $imageDir/ 768 768 $lambda $segDir/

# seg2bbox
python seg2bbox.py $segDir/ $bboxDir/ 200

# evaluation protocol
python Deteval.py $bboxDir $gtDir
