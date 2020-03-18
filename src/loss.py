import tensorflow as tf
import numpy as np

# classification loss object to calculate the classification loss.
classification_loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True)

# Regression loss object to calculate the regression loss.
# TODO: Change the loss as per the experimental results.
regression_loss_object = tf.keras.losses.MSLE


def custom_loss(model, x, y, training, curr_timeStep, final_timeStep):
    """
     A custom loss function for training the model as we want the model to generalize for two things.
        1. Default/ None-default prediction.
        2. Time before the default happened.
    This way, we can generalize for both better prediction and prediction before the actual default happens.

    Arguments:-
     * model: Model which is being trained.
     * x: Inputs
     * y: Outputs
     * training: Bool to indicate of training is going on.
     * curr_timeStep: Current time step.
     * final_timeStep: Time step when default happened. It will be -1 if company never went to deffault.

    Returns the final loss computed. For time being, final loss is one third of classifcation loss and two third of regression loss.
    """
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    # 1. Compute the classification loss.
    y_ = model(x, training=training)
    cls_loss = classification_loss_object(y_true=y, y_pred=y_)

    # 3. Based on whether the company went default or not decide how much
    # weightage we should give both losses.
    if final_timeStep > -1:
        final_loss = cls_loss
        return final_loss.numpy()
    else:
        # Compute the regression loss.
        reg_loss = -regression_loss_object(final_timeStep, curr_timeStep)
        final_loss = ((1 / 3) * cls_loss.numpy()) + \
            ((2 / 3) * reg_loss.numpy())
        return final_loss


def grad(model, inputs, targets, curr_timeStep, final_timeStep):
    """
     A gradient tape for watching the gradients of the loss to optimize the model parametres.

     Arguments:-
     * model: Model which is being trained.
     * inputs: Inputs
     * targets: Outputs
     * training: Bool to indicate of training is going on.
     * curr_timeStep: Current time step.
     * final_timeStep: Time step when default happened. It will be -1 if company never went to deffault.

    Returns the final loss computed and gradients computed.
    """
    with tf.GradientTape() as tape:
        loss_value = custom_loss(
            model,
            inputs,
            targets,
            training=True,
            curr_timeStep=curr_timeStep,
            final_timeStep=final_timeStep)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)
