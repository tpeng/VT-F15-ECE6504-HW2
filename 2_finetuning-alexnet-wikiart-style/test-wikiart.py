import caffe
import numpy as np


caffe_root = './caffe'


MODEL_FILE = 'caffe/models/wikiart/deploy.prototxt'
PRETRAINED = 'caffe/models/wikiart/wikiart_iter_47010.caffemodel'

caffe.set_mode_gpu()

net = caffe.Classifier(MODEL_FILE, PRETRAINED,
               mean=np.load('caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1),
               channel_swap=(2,1,0),
               raw_scale=255,
               image_dims=(227, 227))

def caffe_predict(path):
        input_image = caffe.io.load_image(path)
        #print path
        #print input_image
        prediction = net.predict([input_image])


        #print prediction
        #print "----------"

        #print 'prediction shape:', prediction[0].shape
        #print 'predicted class:', prediction[0].argmax()


        proba = prediction[0][prediction[0].argmax()]
        ind = prediction[0].argsort()[-5:][::-1] # top-5 predictions


        return prediction[0].argmax(), proba, ind

correct = 0
total = 0
for line in open('/home/ubuntu/caffe/models/wikiart/data/wikiart/test.txt'):
   path, label = line.strip().split()
   pred, _, _ = caffe_predict(path)
   print label, pred
   total += 1
   if int(label) == pred:
      correct += 1
print total, correct
