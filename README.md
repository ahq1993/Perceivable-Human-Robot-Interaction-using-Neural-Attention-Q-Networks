# Show, Attend and Interact: Perceivable Human-Robot Social Interaction through Neural Attention Q-Network
This project proivdes the implementation of Multimodal Deep Attention Recurrent Q-Network (MDARQN) using which the robot learned to interact with people in a perceivable and socially acceptable manner after 14 days of interacting with people in public places.

Requirements:

1- Pepper Robot (https://www.ald.softbankrobotics.com/en/cool-robots/pepper)

2- Mbed microcontroller (https://developer.mbed.org/platforms/mbed-LPC1768/)

3- FSR Touch sensor (https://www.sparkfun.com/products/9376)

Procedure:

1- Bring FSR into action through integration with mbed or any microcontroller (please burn the code 'uC.cpp' on mbed).

2- Paste FSR on Pepper's right hand (as shown in fig 1 in [1]).

3- Run mbed_usb.py, robot_listen.py and train_ql.lua on three seperate terminals for data_generation_phase.

4- Run train_ql.lua only for learning_phase.

[1] Ahmed Hussain Qureshi, Yutaka Nakamura, Yuichiro Yoshikawa and Hiroshi Ishiguro, "Show, Attend and Interact: Perceivable Human-Robot Social Interaction through Neural Attention Q-Network", arXiv:1702.08626 [cs.RO] (to appear in IEEE ICRA'17). 
