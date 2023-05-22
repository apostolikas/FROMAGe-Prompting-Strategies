# Exploring in-context learning abilities of Fromage 🧀
28 Mar 2023 | Nikolaos Apostolikas, Panagiotis Tsakas, Vasileios Vythoulkas, Bram Slangen, Denny Smit

---------------

In this blog post, we will discuss the paper "Grounding Language Models to Images for Multimodal Generation". The paper proposes a method to ground pretrained text-only language models to the visual domain, enabling them to process and generate arbitrarily interleaved image-and-text data. 

The goal of this blog post is :

1. Provide an overview of the original paper.
2. Replicate the results (wherever applicable) in order to verify the claims of the authors.
3. Discover the in-context abilities of the model with thorough cross-domain experiments.

#
## Introduction
<!---Humans can learn a new task without requiring huge task-specific supervised datasets. -->

In recent years, the domains of computer vision and natural language processing(NLP) have witnessed the emergence of large-scale models. These models have a vast number of parameters and are pre-trained on huge datasets to acquire extensive knowledge across domains. This development has opened up new possibilities to explore the abilities of these models when few training data are available and without the need to update any of the model's parameters.
<!--- I need to add a smooth transition here -->

<!--- Maybe we can put this image later when we describe the image classification task and put here an image from the image captioning task.For  image captioning it is more clear that it needs both image and text models. 
-->
This is where In-context learning comes up. In-context learning or priming leverage additional context added to the input, which guide the model towards the required result without requiring any gradient updates. A common approach is to add input-label pairs, also known as demonstrations, together with a task instruction as a natural language prompt to an evaluation example. One example of in-context learning is illustrated in the figure above. <br>

![](images_report/ICL.png)

In-context learning seems very appealing because it reduces the need for task-specific data. Hence, zero-shot and few-shot learning can be used. Additionally, no parameters are updated so catastrophic forgetting cannot occur and we can use the same model for multiple tasks. Furthermore, by employing in-context learning in an interface even inexperienced users could easily use AI systems. <br>
Despite its intriguing properties, the models may be sensitive to the prompt that is added to the input. Therefore, exploration of prompting strategies is useful to improve the performance of large models. We will explore the in-context learning abilities of Fromage. <br>

---------
Fromage model
==================================

Model Architecture
---------
![](images_report/fromage_architecture.png)

First, let’s review the model architecture. Fromage combines a vision encoder and a decoder language model while keeping their parameters fixed. Specifically, it employs the CLIP model as a vision encoder and OPT as a language model to be able to handle multimodal data. To map the visual space into the text space and vice versa, learnable linear layers are utilized. Fromage has been trained on the Conceptual Caption dataset [[1]](#cc3m) containing 3.3 million image-text pairs for image-captioning and image-text retrieval. The original paper utilized this dataset for the tasks of image captioning and image-text retrieval.

<!--- I am not sure whether talking about Conceptual Caption is a 
good idea because of the image-captioning dataset. Besides, 
we need to add a picture here
-->

--------------------------

Historical Review
-----------------
There are several vision-language models desrcibed in the literature. Models such as Clip and ALIGN use vision and text encoders and calculate the similarity between the different modalities representations. <!--- Nonetheless, these models are restricted to cases where pre-defined labels are available.--> Other models like our model, Fromage, differ by combining a vision encoder with a text decoder. This allows them to generate text and be used for more open-ended tasks like image-captioning. Fromage in contrast to other models like Flamingo is also able to generate images from the Conceptual Caption Dataset on which it was trained. <br>
<!---Another important distinction between different vision-language models is the way they bridge different modalities. Existing approaches include finetuning cross-attention layers (Flamingo), only vision encoders (Frozen), only text  lightweight transformer blocks (Blip2), directly feeding the   -->
In-context learning became known with the remarkable success of the GPT-3 model in text tasks. Lately, in-context learning has been applied to both the vision-only models and vision-language models. A popular technique to boost the performance of in-context learning is demonstration selection by image or text retrieval. Other techniques include instruction tuning first on other datasets, making the LM generate the prompt, changing the order of the demonstrations or changing the instructions given to the model.

---------------


# Results

The approach followed was to first replicate the results of the paper and then explore the possibilities of in-context learning of the model, by working with different prompting strategies. Although the replication of the results was virtually impossible due to the need of manual annotators, we came up with a workaround to verify whether the results for the datasets used are accurate. Next, we tried to explore the in-context learning potential of FROMAGe, by evaluating it on several tasks (e.g. Video Captioning, Visual Question Answering, etc.). Last but not least, several prompt augmentation methods were used to explore different prompting strategies in order to determine the importance of the input for the model's output.

For simplicity, the table below provides an overview of the experiments conducted.

|             Task            	|      Dataset      	| Samples 	|                           Goal                          	|
|:---------------------------:	|:-----------------:	|:------------------:	|:-------------------------------------------------------:	|
|     Visual Storytelling     	|       VIST        	|                    	|             Verify the claims of the authors            	|
|       Image Captioning      	| Cropped Flickr-8k 	|         214        	| Explore the effect of visual augmentation of the prompt 	|
| Image Retrieval (from text) 	| Cropped Flickr-8k 	|         214        	|  Explore the effect of text augmentation of the prompt  	|
|     Image Classification    	|   mini-Imagenet   	|                    	|    Remove the recency bias with visual augmentations    	|
|       Video Captioning      	|       TGIF        	|                    	|       Explore the in-context abilities of FROMAGe       	|
| Visual Question Answering     |    Guided-VQA        	|         300          	| Reveal possible limitations of ICL & prompt augmentations |
-----------------


## Reproducibility


Two of the main experiments conducted by the authors were: 
1. Image retrieval from visual and text input (Visual Storytelling).
2. Text retrieval from visual and text input (Visual Dialog). 

The first experiment could be reproduced up to a point. On the contraty, the second experiment could not be replicated. That is because the authors mention that they computed the perplexity of each question and answer sequence to measure performance, but they do not give enough details on how the needed probabilities were obtained. Therefore, this experiment was skipped due to lack of information about the perplexity scores.

-----------------


## Experiment 1: Image retrieval from visual and text input


The authors assessed the performance of FROMAGe in retrieving the appropriate image conditioned on a sequence of interleaved image-text inputs from the Visual Storytelling (VIST) dataset [[?]](#vist). One example of a story sample from the dataset is shown bellow. 

![](images_report/vist-story-from-paper.png)

The goal of this experiment was to observe the performance of FROMAGe in image retrieval when more context was provided as input. The first setting provided only the last sentence of the dialog and the last setting provided the whole dialog (sentences and images except the last image of course).

From those different settings, it was observed that the performance in image retrieval increased when more context was given as input. A lot of manual annotation was necessary for the evaluation of this experiment, which was infeasible in the context of this project.

![](images_report/vist-trend-from-paper.png)

To work around this problem, the only possible solution was to make our own manual annotations on 100 random samples. Based on our judgment, we would annotate if the retrieved image was good or not. To be able to evaluate our annotations we used accuracy as metric, which would allow us to observe the trend of the performance when more context was provided as input.

Our results are shown the table bellow. (The missing accuracy is going to determine whether we confirm the upward trend the authors report or not.)

| Input                | Accuracy |
| -------------------- | -------- |
| 1 caption            |   ???    |
| 5 captions, 4 images |   35%    |

-----------------
&nbsp;

## Extension 
-----------------
Our extension can be divided into two parts. The first part is to explore in depth the in-context abilities of FROMAGe. This will be done by using new tasks and datasets. Specifically, in-context learning aims to make the model able to perform a task just by conditioning on input-output examples, without updating the actual parameters. In simple words, we will first give some input-output examples to the model, so it understands the task and then we will evaluate it on the query example. The second part of the extension is to prove whether prompt augmentations (visual and text) can lead to enhanced performance for the new tasks.

-----------------
&nbsp;


### Image Captioning


Although the model was trained on the CC3M dataset, it is useful to check how it performs on other datasets as well. For this purpose, we used the Flickr-8k dataset, from which we used a specific subset that according to experts, the captions are fully representative of the corresponding image. Furthermore, we augmented the input visually by adding more images. This means that given the original image that the model needed to caption, we instead asked the model to retrieve 2 similar images. After retrieving the similar images, we added them to the prompt and asked the model to perform Image Captioning for the original image. Simply put, instead of giving directly the input image, we retrieved 2 similar ones and gave all three as input, but only asked the model to caption the original one. To evaluate the model, we employed a language model to compute the text embeddings of the generated caption using visual augmentation, the generated caption without using any augmentation and the original caption.  

![](images_report/Visual_augmentation_of_prompt.png)

After obtaining the embeddings, the cosine similarity was computed using each token in the text output of the model with each token in the original caption for both cases (i.e. using visual augmentation or not). The results for this experiment can be seen in the following table.

|  Prompt  	| Cosine Similarity 	|
|:---------:	|:-----------------:	|
| With augmentation 	|         0.478         |
| No augmentation 	    |         0.343         |

It can be seen from the table that giving some similar examples along with the query image leads to the model generating a representative caption of the original image. This means that the generated caption using visual augmentation is closer to the original caption, which serves as a target.

-----------------
&nbsp;


### Image Retrieval from Text 

In this task, we used the Flickr-8k dataset, by giving the model the caption as input and asking it to retrieve a similar image from the CC3M dataset. Moreover, for this experiment, we augmented the text input by expanding the caption. This was done by prompting the GPT-3 model asking it to provide more information about each caption. Specifically, we asked it to add more descriptive words (query expansion). Our goal was to check whether the augmented text input will make the model retrieve a better image than the one retrieved by the original caption. The following figure explains the aforementioned procedure.

![](/images_report/Text_augmentation_of_prompt.png)

As an evaluation metric, cosine-similarity was used to compare the visual embeddings of the retrieved image using the original caption and the target image. The same was applied for the retrieved image using the augmented caption. The final step was to compare whether the cosine similarity was higher with the augmented caption or not. The visual embeddings were obtained by using the CLIP component of the FROMAGe model. The cosine-similarity displayed below is the average for all the samples seen by the model.

|  Caption  	| Cosine Similarity 	|
|:---------:	|:-----------------:	|
|  Original 	|         0.67        	|
| Augmented 	|         0.70         	|

Looking at the results table above, the conclusion is that in most cases, a text augmentation of the input can actually help me the model retrieve a better image (i.e. more similar to the target image).

-----------------

### Image classification

|Model |Accuracy|
|-----|--------|
|Frozen|33.7      |
|Fromage  |35.56      |


We also evaluated our model on the mini-Imagenet dataset. Specifically, we worked on the few-shot setting where we add to the input two demonstrations -one with the correct label and another with a different label. As shown in the table above, the model's performance in this setting was poor, similar to what reported in the Frozen paper. We observed that the model suffers from recency bias (cite), meaning it almost always predict the label of the demonstration that is closest in proximity to the test input. (We plan to apply visual augmentation here as well)

-----------------
&nbsp;



### Visual Question Answering

In this task, we used the guided vqa dataset (cite). A sample consists of two pairs of images-captions, a question, a question image and the answer to the question. It is found that the model struggles to perform well in this task. This is due to the fact that some of the questions refer to secondary objects of the image or objects in the background, this making the task a bit tricky. A simple solution seemed to be to segment the query image and then add it to the prompt. For this visual augmentation of the prompt, we employed the CLIPSeg model and the Oneformer model. A demonstration of the above can be seen in the following figure.

![](/images_report/gvqa.png)

After obtaining the generated captions for each case, we compared them to the original answer of the sample. In order to achieve this, we used the cosine similarity metric of the text embeddings generated by the MiniLM-L6-v2 by giving as input the generated captions from FROMAGe.

|  Prompt  	| Cosine Similarity 	|
|:---------:	|:-----------------:	|
|  Original 	|        0.296   	|
| Augmented using CLIPSeg	|         0.314       	|
| Augmented using Oneformer	|         0.280       	|

As the results suggest, in the case of the CLIPSeg, the augmented prompt enhances the performance, but not significantly. On the contrary, the additional segmented image by Oneformer in the prompt does not seem to help the model. This task proves that in-context learning can not be always applied effectively for every task and also reveals the limitations of prompt augmentations.

&nbsp;


### Insights


1.  
    
    
2.  
    

Conclusion
==========

In a nutsell, since FROMAGe does not employ extremely large models, other methods and strategies had to be explored to enhance its performance on several tasks. Regarding the goals of this project, the possibilities for in-context learning of the model were explored in depth for various tasks. Furthermore, different prompting templates and strategies, such as visual and text augmentation of the prompt have proven to be beneficial for the model, since its performance in all  cases was increased. Lastly, it is important to understand the advantages of in-context learning, where we do not update the parameters of the model, but it's also crucial to understand through the experiments conducted how the prompt itself and different prompting strategies play a significant role for the performance of the model.


References
==========

<a id="cc3m"></a> [1] P. Sharma, N. Ding, S. Goodman, R. Soricut, Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning, in: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Association for Computational Linguistics, Melbourne, Australia, 2018: pp. 2556–2565. https://doi.org/10.18653/v1/P18-1238.

    