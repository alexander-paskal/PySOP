import os, sys
sys.path.append(os.getcwd())
import click


@click.group()
def cli():
    pass


@cli.command()
def gradient_descent():
    from descent_methods.gradient_descent import main
    main()




if __name__ == '__main__':
    cli()
