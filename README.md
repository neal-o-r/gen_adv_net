# Generative Adversarial Network

A 1-D generative adversarial network. Comprises a generator which takes in an *x* value and returns a *y* (see [Universal Function Approximator](https://github.com/neal-o-r/function_approx)), and an adversary which takes *(x,y)* tuples and returns a binary classification. These parts a linked by a referee, which will produce a random *x*, and then either feed it through the generator or some set function *f(x)*. It will then give the resultant *(x,y)* to the adversary. The adversary aims to correctly distinguish the function from the generator as often as possible, the generator aims to trick the adversary. And so they both learn representations  of the hidden funciton *f(x)*. In the figure below is a function *f(x)*, values from the generator (marked with squares) and coloured by classification (green-right, red-wrong).

![results](https://github.com/neal-o-r/gen_adv_net/blob/master/results.png)

