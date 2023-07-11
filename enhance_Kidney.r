## Enhance tutorial: Applied pre-trained DeepGWAS model to other GWAS data
# 11.02.2022
# Gang Li

library(keras);
library(stringr);
library(pbapply);
library(tensorflow);

## prepare the data to be enhanced
example_data <- read.table('/proj/yunligrp/users/theowu/DeepGWAS/data/Wuttke.2019.features.all.selected.txt',header=T)

dim(example_data) # 47386    35
head(example_data)

select_list<-2:22
pred_data <- example_data[,select_list];
pred_data = data.matrix(pred_data)

dim(pred_data) # 9958786      34    
head(pred_data)


## load the model 

model_dir = "/proj/yunligrp/users/shuaihuang/DeepGWAS/DeepGWAS/"
model_name = "DeepGWAS_Kidney_Model.h5"

model <- keras_model_sequential();

model %>% 
layer_dense(units = 5, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(length(select_list))) %>% 
layer_dense(units = 20, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(5)) %>% 
layer_dense(units = 5, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(20)) %>% 
layer_dense(units = 10, activation = 'relu',kernel_initializer = 'orthogonal',input_shape = c(5)) %>% 
layer_dense(units = 5, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(10)) %>% 
layer_dense(units = 10, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(5)) %>% 
layer_dense(units = 5, activation = 'relu',kernel_initializer = 'orthogonal',input_shape = c(10)) %>% 
layer_dense(units = 10, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(5)) %>% 
layer_dense(units = 5, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(10)) %>% 
layer_dense(units = 10, activation = 'relu',kernel_initializer = 'orthogonal',input_shape = c(5)) %>% 
layer_dense(units = 5, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(10)) %>% 
layer_dense(units = 10, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(5)) %>% 
layer_dense(units = 5, activation = 'relu', kernel_initializer = 'orthogonal',input_shape = c(10)) %>% 
layer_dense(units = 1, activation = 'sigmoid', kernel_initializer = 'orthogonal',input_shape = c(5))

model %>% compile(
loss = 'binary_crossentropy',
optimizer = optimizer_adam(lr = 0.0005),
metrics = 'accuracy'
)

model <- load_model_hdf5(paste0(model_dir,model_name))

## predict
pred_gang <- model %>% predict(data.matrix(example_data[,select_list]))

# colnames(pred_data)[32] = 'log10p'
# observed = ifelse( as.numeric(pred_data$log10p) > -log10(5e-08) ,1,0);
# pred_test_label = ifelse(pred_gang > 0.5,1,0);
# table2 = table(observed,factor(pred_test_label,levels=0:1));

# print("Enhanced results:")
# print(table2)
# FN = intersect( which(pred_test_label == 0), which(oberserved_third == 1) )

head(pred_gang)
write.table(pred_gang,"DeepGWAS_enhanced_prob.txt",quote = F,sep="\t",col.names=T,row.names = F)

### END ####