# chess-probing

This is a project evaluating the performance of a model with GPT-2 architecture trained on chess games with moves in the form `(start square, end square)`. Using the pretrained model from [Tosniwal et al. (2021)](https://github.com/shtoshni92/learning-chess-blindfolded), whose work asks whether or not language models can perform state-tracking, I evaluate the cases from their test data where the language model fails at state-tracking, and attempt to outline why these failures occur.

The full write-up of this project can be found [here](https://sweeneyjessica.github.io/linguistics/2021/04/10/chess-state-tracking.html).
