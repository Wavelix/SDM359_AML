- Weight is obtained by sampling from the standard normal distribution and multiplying by $0.01$

- Activation function: hyperbolic tangent function

$$
\delta^z(k)=d(k)-z(k)
$$

$$
\delta^y_i(k)=(d(k)-z(k))v_i(k)\Phi'(y_i(k))
$$

$$
\frac{\partial E}{\partial v_i}=(d(k)-z(k))\cdot (-y_{fi}(k))=-\delta^z(k)\cdot y_{fi}(k)
$$

$$
\frac{\partial E}{\partial w_{il}}=(d(k)-z(k))(-v_i(k))\Phi'(y_i(k))x_l(k)=-\delta^y_i(k)x_l(k)
$$
