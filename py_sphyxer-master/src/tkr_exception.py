

class TkrException:

    def __init__(self):
        pass


class TkrSymbolNotFound(TkrException):

    def __init__(self):
        super().__init__()

    print('Symbol not found')


class TkrDataNotFound(TkrException):

    def __init__(self):
        super().__init__()

    print(f'Historic data not found')
