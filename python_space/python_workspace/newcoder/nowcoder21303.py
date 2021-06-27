class nowcoder21303:

    def delete_brakets(self, s: str, target: str) ->str:
        if s == target:
            return "Possible"
        count = 0
        while '()' in s:
            s.replace()
            if s == target:
                return "Impossible"
