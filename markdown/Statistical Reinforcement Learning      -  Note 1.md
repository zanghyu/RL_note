# Statistical Reinforcement Learning      -  Note 1

## MDP

### Value and Policy

   	Boundedness of **rewards** :   $r_{t} \in\left[0, R_{\max }\right]​$ 

​	   Boundedness of $\mathbb{E}\left[\sum_{t=1}^{\infty} \gamma^{t-1} r_{t}\right]​$ :       $\mathbb{E}\left[\sum_{t=1}^{\infty} \gamma^{t-1} r_{t}\right] \in [0, \frac{R_{\max }}{1-\gamma}]​$ 

​						-Reason:    等比级数： $\sum_{n=0}^{\infty} a q^{n}(a \neq 0)$ 

​											当$0<|q|<1$时，$\sum_{n=0}^{\infty} a q^{n}$收敛，且收敛于$\frac{a}{1-q}$ 

​	  Define  $V^{\pi}(s)=\mathbb{E}\left[\sum_{t=1}^{\infty} \gamma^{t-1} r_{t} | s_{1}=s, \pi\right]​$ 

​		So, $V^{\pi}(s) < \frac{R_{max}}{1-\gamma}$

### Policy evaluation

<img src="../picture/Statistical Reinforcement Learning_Note1/01.png" style="zoom:50%"/>

<img src="../picture/Statistical Reinforcement Learning_Note1/02.png" style="zoom:50%"/>

​                									<img src="../picture/Statistical Reinforcement Learning_Note1/03.png" style="zoom:50%"/>

<img src="../picture/Statistical Reinforcement Learning_Note1/04.png" style="zoom:50%"/>

