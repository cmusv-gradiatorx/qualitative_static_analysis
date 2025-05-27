AUTOGRADER REQUIREMENTS

The project requirement is to create an autograder for graduate student level assignments in software engineering.
This project's scope is to do a qualitative level analysis of the entire code using LLMs.

The idea is to provide the LLM the entire project detail and code using repomix(documentation: https://repomix.com/guide/), also the project rubric and expect a response with remarks, suggestions, improvement scope and grades.
Every violation of the rubric in the code base needs to be pointed out.

For the codebase, the limitation is that the entire code base can go over the token limit of the LLM being used.
There is a requirement of some tool for counting the tokens. For the code base repomix has token count. Also if the code base is going over limit, compress functionality of repomix can be used.

The entire codebase should be highly modular and extensible. Use design patterns wherever possible. The code base should be highly maintainable. 
Example:
The LLM used can change over time, so the way its used should be like plug and play. A folder for the implementation of all the LLMs should be there(example gemini, openai, Ollama based llama and deepseek). At the point of usage the used model should be chosen from some varaibale in env file. Then because plug and play, that model's implementation should be used(maybe using class).

Same way all prompts should be in a folder. The prompts I can think of right now is the prompt containing the rubric, instruction for feedback and the overall prompt(which combines the different prompt). The overall prompt should have the rubric prompt, the codebase(after repomix creates the doc: that should be a complex logic by itself, where token count is checked for if the entire codebase can be used or compress, again put that logic in a different file) and the intruction for instruction prompt. Its response should give the complete response, the feedback, the score everything.
For rubric prompt file, it should have a plug in for the content of a text file. This text file will be filled by the instructor to put in the actual prompt. Same way for instruction prompt file, plugin from another text file where the instructor adds in the instructions to do the grading. For now fill these files yourself, I will edit them according to the project.
For final output, no need to go with specific strcuture of output since that will make the system less flexible. The output should just be a text output as defined in the instruction prompt. later if needed specific structure can be added there. So no need to create specific parsers as well. Just get the final output.


For the repomix logic, go through the docs and see how best it can be used in the project, with token logic, compress and everything. For the repomix input folder, expect it from a zip file presnt in a folder(create the folder). I will add the codebase as zip to that folder(donot unzip yourself, check repomix docs, it takes in zips). At the end of processing delete this zip file.

For the start atleast provide suppoert for gemini, openai, llama from ollama. Keep the code highly modular and extensible, maintainability is of peak importance. Use design patterns where needed, follow OOPS if that helps in extensibility. Comment code like a proper software engineer. Do not create extra class/functions if not needed. The project should follow best coding practices but be simple. In future we might have a lot of extra stuff so doing extra thing right now might make it complex (for example, no need to keep stuff like batch processing right now, if needed we can do those later. Right now just have an input folder to take zip input and an output folder to dump the final LLM response). Language of coding should be python. Create the env file as well with dummy creds, i will fill. For the config, I will do that using env, no need to create a cli based solution.

NEW FEATURE

The above requirement is for a core service:autograder
Next service: A static analysis on basic of injectable rules on sem grep. Again following best coding practices for maintainability extensibility and design patterns.
What this is for for: The zip file frovided as input is fed into semgrep(with whatver preprocessing is needed). The semgrep works on rules in my understanding. The rules should be injectable as those will change from assignment to assignment:
One example you can add in the rules file for now:

rules:
  # Single Responsibility Principle (SRP) Violations
  - id: srp-large-class
    pattern-either:
      - pattern: |
          class $CLASS {
            ...
            $METHOD1(...) { ... }
            ...
            $METHOD2(...) { ... }
            ...
            $METHOD3(...) { ... }
            ...
            $METHOD4(...) { ... }
            ...
            $METHOD5(...) { ... }
            ...
            $METHOD6(...) { ... }
            ...
            $METHOD7(...) { ... }
            ...
            $METHOD8(...) { ... }
            ...
          }
    message: "Class may violate SRP - consider breaking down large classes with many methods"
    languages: [java, csharp, typescript, javascript]
    severity: WARNING

The result from the semgrep analysis on the codebase on the basis of rules defined(again, keep this configurable, app needs to be extensible) should be passed to a LLM (use existing LLM provider, add a new prompt to prompts folder, so that too is configurable). For now, write in that prompt to provide an analysis of the static analysis. Use the rubric and create a new prompt file called as static_instructions. In the final file being generated which only had autograder stuff before, append the LLM repsonse for static analysis using semgrep as well. Don't change anything in the autograder, add this new component. And also as before: Use design patterns where needed, follow OOPS if that helps in extensibility. Comment code like a proper software engineer. Do not create extra class/functions if not needed. The project should follow best coding practices but be simple. In future we might have a lot of extra stuff so doing extra thing right now might make it complex (for example, no need to keep stuff like batch processing right now, if needed we can do those later). Language of coding should be python.