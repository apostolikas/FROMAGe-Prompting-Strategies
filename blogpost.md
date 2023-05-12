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

First, let’s review the model architecture. Fromage combines a vision and a language model while mainting their parameters fixed. To map the visual space into the text space and vice versa, learnable linear layers are utilized. Fromage is trained on the Conceptual Caption dataset (cite) containing 3.3 million image-text pairs. 
<!--- I am not sure whether talking about Conceptual Caption is a 
good idea because of the image-captioning dataset. Besides, 
we need to add a picture here
-->

--------------------------


Historical Review
-----------------



Results
-----------------



### Insights


1.  
    
    
2.  
    

Conclusion
==========


References
==========


    