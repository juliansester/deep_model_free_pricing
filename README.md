# A deep learning approach to data-driven model-free pricing and to martingale optimal transport

Ariel Neufeld and Julian Sester.

## Abstract
We introduce a novel and highly tractable supervised learning approach based on neural networks that can be applied for the computation of model-free price bounds of, potentially high-dimensional, financial derivatives and for the determination of optimal hedging strategies attaining these bounds. In particular, our methodology allows to train a single neural network offline and then to use it online for the fast determination of model-free price bounds of a whole class of financial derivatives with current market data. We show the applicability of this approach and highlight its accuracy in several examples involving real market data. Further, we show how a neural network can be trained to solve martingale optimal transport problems involving fixed marginal distributions instead of financial market data.


## Usage
The Folder 'Example 2.5' contains the python notebook related to Example 2.5 and the trained neural network as *.h5-file.
The Folder 'Example 2.6' contains the python notebook related to Example 2.6 and the trained neural network as *.h5-file.
The Folder 'Example 2.7' contains the python notebook related to Example 2.7 and the trained neural network as *.h5-file.
The Folder 'Examples MOT' contains the python notebook related to Example 3.5, Example 3.6, 
the files needed to create the samples (as *.py files) and the trained neural network as *.h5-file.


## License
MIT License

Copyright (c) 2021 Julian Sester

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.