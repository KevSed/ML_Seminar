from plot_history import plot_history
from keras.models import load_model
import click

@click.command()
@click.argument('model', type=click.Path(exists=True))
@click.argument('name', required=True, type=string)
def main(model, name):
    hist = model[]
    plot_history(hist)


if __name == '__main__':
    main()
