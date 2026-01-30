class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x):
        x = self.model.conv(x)
        x.register_hook(self.save_gradient)  # Register the hook for the target layer

        x = self.target_layer(x)
        x = x.view(x.size(0), -1)

        x = self.model.fc(x)
        return x