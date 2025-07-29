This model is based on the FER-2013 dataset for facial emotion recognition. 
Due to the low number of images, especially regarding the emotion 'disgust' there is further augmentation of this train/test file

Accuracy log:
    250727fer_model.pth: mod test=58.23380992430614% 
    250727fer_model.pth v.2: mod test = 69.20100925147183%
    250727fer_model.pth v.3: mod test = 73.57494299771992%


raf-db augmentation:
    due to the low number of images we augmented the database 

so far the model is trained on fer-2013 and raf-db.
for evaluation purposes we downloaded a collection of different databases from kaggle.
https://www.kaggle.com/datasets/tuna3686/fer-dataset
We used two smaller databases for evaluation (CK and AffectNet_val)

improve model by adding more images for disgust: https://www.sciencedirect.com/science/article/abs/pii/S0005796716301978#preview-section-cited-by
https://zenodo.org/records/167037
copied images to dataset2, so i dont have to change the image- and dataloader