import predict_utils
import torch


def main():
    # get command line arguments
    in_arg = predict_utils.get_input_args()
    
    # process image
    print('Processing image...')
    image = predict_utils.process_image(in_arg.image_path)
    print('Image processed\n')
    
    # load model
    print('Loading saved model...')
    model = predict_utils.load_model(in_arg.checkpoint_path)
    print('Model loaded\n')
    
    print('Predicting...')
    with torch.no_grad():
        if in_arg.gpu == True:
            model.to('cuda')
            output = model.forward(image.cuda())
            model.to('cpu')
        else:
            output = model.forward(image)
    print('Prediction finished\n')
    
    if in_arg.gpu == True:
        probes, classes = torch.exp(output.cpu()).topk(in_arg.top_k)
    else:
        probes, classes = torch.exp(output).topk(in_arg.top_k)
    
    probes = probes.detach().numpy().tolist()[0]
    indices = classes.detach().numpy().tolist()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    cat_to_name = predict_utils.load_category_names(in_arg.category_names)
    flowers = [cat_to_name[c] for c in classes]
    
    for i in range(len(probes)):
        print('{}: {:.2f}%'.format(flowers[i], probes[i]*100))



if __name__ == "__main__":
    main()