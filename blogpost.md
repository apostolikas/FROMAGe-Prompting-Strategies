Exploring in-context learning abilities of Fromage 🧀
================================================================

28 Mar 2023 | Nikolaos Apostolikas, Panagiotis Tsakas, Vasileios Vythoulkas, Bram Slangen, Denny Smit

<!---Humans can learn a new task without requiring huge task-specific supervised datasets. -->

In recent years, the domains of computer vision and natural language processing(NLP) have witnessed the emergence of large-scale models. These models have a vast number of parameters and are pre-trained on huge datasets to acquire extensive knowledge across domains.
<!--- I need to add a smooth transition here -->

![](images_report/lion_malamute.png)

<!--- Maybe we can put this image later when we describe the image classification task and put here an image from the image captioning task.For  image captioning it is more clear that it needs both image and text models. 
-->
In-context learning or priming (cite) leverage additional context added to the input, which guide the model towards the required result without requiring any gradient updates. A common approach is to add input-label pairs, also known as demonstrations, together with a task instruction as a natural language prompt to an evaluation example. <br>
In-context learning seems very appealing because it reduces the need for task-specific data. Hence, zero-shot and few-shot learning can be used. Additionally, no parameters are updated so catastrophic forgetting cannot occur and we can use the same model for multiple tasks. Furthermore, by employing in-context learning in an interface even inexperienced users could easily use AI systems. <br>
Despite its intriguing properties, the models may be sensitive to the prompt that is added to the input. Therefore, exploration of prompting strategies is useful to improve the performance of large models. We will explore the in-context learning abilities of Fromage. <br>

Fromage model
==================================

Model Architecture
---------

First, let’s review the model architecture. Fromage combines a vision and a language model while mainting their parameters fixed. To map the visual space into the text space and vice versa, learnable linear layers are utilized. Fromage is trained on the Conceptual Caption dataset [[1]](#cc3m) containing 3.3 million image-text pairs. 
<!--- I am not sure whether talking about Conceptual Caption is a 
good idea because of the image-captioning dataset. Besides, 
we need to add a picture here
-->

--------------------------


Historical Review
-----------------



Results
-----------------

The approach followed was to first replicate the results of the paper and then explore the possibilities of in-context learning of the model, by working with different prompting strategies. Although the replication of the results was virtually impossible due to the need of manual annotators, we came up with a workaround to verify whether the results for the datasets used are accurate. Next, we tried to explore the in-context learning potential of FROMAGe, by evaluating it on several tasks (e.g. Video Captioning, Visual Question Answering, etc.). Last but not least, several prompt augmentation methods were used to explore different prompting strategies in order to determine the importance of the input for the model's output.
<!--- Not sure where we should put these paragraphs -->

Image Captioning

Although the model was trained on the CC3M dataset, it is useful to check how it performs on other datasets as well. For this purpose, we used the Flickr-8k dataset, from which we used a specific subset that according to experts, the captions are fully representative of the corresponding image. Furthermore, we augmented the input visually by adding more images. This means that given the original image that the model needed to caption, we instead asked the model to retrieve 2 similar images. After retrieving the similar images, we added them to the prompt and asked the model to perform Image Captioning for the original image. Simply put, instead of giving directly the input image, we retrieved 2 similar ones and gave all three as input, but only asked the model to caption the original one. To evaluate the model, we used the BertScore metric, which compared the model's generated caption with the target. 

Image Retrieval from Text 

In this task, we used the Flickr-8k dataset, by giving the model the caption as input and asking it to retrieve a similar image from the CC3M dataset. Moreover, for this experiment, we augmented the text input by expanding the caption. This was done by prompting the GPT-3 model asking it to provide more information about each caption. Our goal was to check whether the augmented text input will make the model retrieve a better image than the one retrieved by the original caption. As an evaluation metric, we used ClipScore.


### Insights


1.  
    
    
2.  
    

Conclusion
==========


References
==========

<a id="cc3m"></a> [1] P. Sharma, N. Ding, S. Goodman, R. Soricut, Conceptual Captions: A Cleaned, Hypernymed, Image Alt-text Dataset For Automatic Image Captioning, in: Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Association for Computational Linguistics, Melbourne, Australia, 2018: pp. 2556–2565. https://doi.org/10.18653/v1/P18-1238.

    