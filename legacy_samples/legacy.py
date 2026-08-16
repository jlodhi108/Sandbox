import os


def greet(name):
    return "Hello, %s!" % name


def read_config_path(base_dir, filename):
    return os.path.join(base_dir, filename)


def main():
    print(greet("world"))
    print(read_config_path("/etc", "config.ini"))


if __name__ == "__main__":
    main()
