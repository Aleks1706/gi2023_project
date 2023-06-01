# Project description
Train the machine learning model that predicts cell annotations using gene expression and spatial coordinates on Mouse embryo 9.5 and Mouse brain spatial transcriptomic samples. The samples can be read to the Anndata object using the Scanpy library. After training, the model should be used to refine (correct) annotations for the cells. The proposed approach makes a stratified-by-annotation K-fold split of the training instances, trains the model K times on K-1 folds, and makes predictions for the cells in the remained fold. If some cell obtains more than L out of K same predictions that are different than its initial annotation then its annotation should be changed (refined) to this new annotation. Other approaches for cell annotation refinement using a trained model could be used instead of the proposed one. Feature selection for training the model is challenging due to large number of genes, so some of the dimensionality reduction algorithms such as PCA, UMAP, or some statistical methods from sklearn.feature_selection.SelectKBest should be used. 
Test different combinations of the number of features, K and L and plot the results in spatial coordinates (scanpy.pl.spatial) for both mentioned datasets. 
Create a PowerPoint (Google Slides) presentation explaining all the work being done and a video presentation publicly available on Youtube. Perform code versioning on the Github repository and provide the link to it. 

# Instructions to run the project
Change the paths to filename_mouse_brain and filename_mouse_embryo.
Otherwise just run the project in the cell order as normal.

You can run the cell with plotting multiple times. It chooses 3 random rows from the table of results to plot so you can see how the image changes for different values of features, folds and l.

# Link to the youtube tutorial
