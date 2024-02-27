class Patcher:
    def __init__(self):
        pass

    def get_num_tuples(self, line_nums):
        output = []
        current_start = None
        for num in line_nums:
            if current_start is None:
                current_start = num
            elif num - 1 != line_nums[line_nums.index(num) - 1]:
                output.append((current_start, line_nums[line_nums.index(num) - 1]))
                current_start = num
        output.append((current_start, num))
        return output

    def stitch(self, prompt_text, lines, line_nums):
        # Store starting adn ending location of the context block
        locations = self.get_num_tuples(line_nums)

        # Check if the context delimiter is removed from the LLM output
        if len(locations) != len(prompt_text):
            # Original file is the output file -> File marked as unfixed.
            print("=============================")
            print("Context delimiter removed")
            print("=============================")
            print(locations)
            for prompt in prompt_text:
                print(prompt)
            print(f"--------len of prompts: {len(prompt_text)} ---------")
            return lines

        # stitch the lines
        lines_updated = []
        for i in range(len(locations)):
            try:
                # uncomment this line for localisation experiment
                # TODO: make the patchFixer interface same for no_localisation and localisation
                # lines_updated = lines[:locations[i][0]] + [prompt_text[i]] + lines[locations[i][1]+1:]
                lines_updated = (
                    lines[: locations[i][0]]
                    + [x for x in prompt_text[i].split("\n")]
                    + lines[locations[i][1] + 1 :]
                )
                adjustment = len(prompt_text[i].split("\n")) - (
                    locations[i][1] - locations[i][0] + 1
                )
            except IndexError:
                # If the context block ends at the end of the file, then locations[i][1]+1 gives an IndexError
                # lines_updated = lines[:locations[i][0]] + [prompt_text[i]]
                lines_updated = lines[: locations[i][0]] + prompt_text[i].split("\n")
                adjustment = len(prompt_text[i].split("\n")) - (
                    locations[i][1] - locations[i][0] + 1
                )
        return lines_updated, adjustment
