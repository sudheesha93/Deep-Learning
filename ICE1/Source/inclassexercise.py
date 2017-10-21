# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 18:37:33 2017

@author: sudhesa
"""


import tensorflow as tf

a=tf.constant(2)
b=tf.constant(3)
c=tf.constant(5)

op3= tf.pow(a,2 )
op4= tf.add(op3,b)
op5= tf.multiply(op4,c)

with tf.Session() as sess :
    op6=sess.run(op5)
    print(op6)
    
    

op7= tf.pow(a,2 )
op8= tf.add(op7,b)
op9= tf.multiply(op8,c)

with tf.Session() as sess :
    op10=sess.run(op9,feed_dict={a:2, b:3,c:5})
    print(op10)
     