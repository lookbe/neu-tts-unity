using LlamaCpp;
using System;
using System.Threading;
using UnityEngine;

namespace NeuTTS
{
    public class NeuTTSModel : Completion
    {
        protected class ClonePayload : CompletionPayload
        {
            public string Prompt;
            public string Transcript;
            public string AudioText;
        }

        public void PromptWithClone(string prompt, string transcript, string audioText)
        {
            if (string.IsNullOrEmpty(prompt) || string.IsNullOrEmpty(transcript) || string.IsNullOrEmpty(audioText))
            {
                Debug.LogWarning("invalid prompt");
                return;
            }

            if (_llamaContext == IntPtr.Zero)
            {
                Debug.LogError("invalid context");
                return;
            }

            if (_llamaModel == IntPtr.Zero)
            {
                Debug.LogError("model not loaded");
                return;
            }

            if (status != ModelStatus.Ready)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Generate;
            var payload = new ClonePayload()
            {
                Prompt = prompt,
                Transcript = transcript,
                AudioText = audioText,

                Temp = this.temperature,
                TopK = this.topK,
                TopP = this.topP,
                MinP = this.minP,
                RepeatPenalty = this.repeatPenalty,

            };

            RunBackground(payload, RunPromptWithClone);
        }

        void RunPromptWithClone(ClonePayload inputPayload, CancellationToken cts)
        {
            string user = $"user: Convert the text to speech:<|TEXT_PROMPT_START|>{inputPayload.Transcript} {inputPayload.Prompt}<|TEXT_PROMPT_END|>";
            string assistant = $"assistant:<|SPEECH_GENERATION_START|>{inputPayload.AudioText}";
            string prompt = $"{user}\n{assistant}";

            int[] prompt_token = Tokenize(prompt);

            var payload = new GenerationPayload()
            {
                Tokens = prompt_token,
                Temp = inputPayload.Temp,
                TopK = inputPayload.TopK,
                TopP = inputPayload.TopP,
                MinP = inputPayload.MinP,
                RepeatPenalty = inputPayload.RepeatPenalty,

            };

            RunGenerate(payload, cts);
        }
        protected override bool EndGeneration(int token, int generated_token_count)
        {
            int[] prompt_token = Tokenize("<|SPEECH_GENERATION_END|>");
            if (prompt_token.Length > 0)
            {
                return Native.llama_vocab_is_eog(_llamaVocab, token) || token == prompt_token[0];
            }
            return Native.llama_vocab_is_eog(_llamaVocab, token);
        }
    }
}
