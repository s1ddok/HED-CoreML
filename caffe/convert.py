import coremltools

coreml_model = coremltools.converters.caffe.convert(('hed_pretrained_bsds.caffemodel', 'deploy_main.prototxt'),
    image_input_names='data',
    is_bgr=True,
    red_bias=122.67891434, green_bias=116.66876762, blue_bias=104.00698793)

coreml_model.author = 'Original paper: Xie, Saining and Tu, Zhuowen. Caffe implementation: Yangqing Jia. CoreML port: Andrey Volodin'
coreml_model.license = 'Unknown'
coreml_model.short_description = "Holistically-Nested Edge Detection"

coreml_model.input_description['data'] = 'Input image to be edge-detected. Must be exactly 500x500 pixels.'
coreml_model.output_description['upscore-fuse'] = 'Tensor that can be treated as single-channel texture where each pixel represents the probability of edge there. Must be normalized before usage (i.e. with sigmoid function)'
coreml_model.save('HED_main.mlmodel')
