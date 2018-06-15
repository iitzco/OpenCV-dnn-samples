OpenCV contiene una libreria `dnn` que soporta inferencias para Caffe, Tensorflow y Torch. En cada una de estas, soporta las operaciones mas comunes (capas densas, convolucionales, etc.)

El blog oficial de la persona que lo implemento:

https://habr.com/company/intel/blog/333612/

Wiki de OpenCV:

https://github.com/opencv/opencv/wiki/Deep-Learning-in-OpenCV

Luego, algun tutorial:

https://www.pyimagesearch.com/2017/08/21/deep-learning-with-opencv/


Algunas notas:

* No soporta GPU por ahora.
* No soporta Caffe2
* Obviamente, al ser OpenCV, va a funcionar mejor en cosas relacionadas a imagenes.
* Algunos modelos de://github.com/facebook/fb.resnet.torch/tree/master/pretrained (menos el 18, que es el que se usa) tuvieron problemas al cargarse.
