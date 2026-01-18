using System.Collections.Generic;
using UnityEngine;

namespace NeuTTS
{
    public class DecoderStreamer : MonoBehaviour
    {
        public NeuTTSModel model;
        public DecoderModel decoder;
        public int streamChunkSize = 480;

        private List<string> streamTokenBuffer = new List<string>();

        private void OnEnable()
        {
            if (model != null)
            {
                model.OnResponseStreamed += OnBotResponseStreamed;
                model.OnResponseGenerated += OnBotResponseGenerated;
            }
        }

        private void OnDisable()
        {
            if (model != null)
            {
                model.OnResponseGenerated -= OnBotResponseGenerated;
                model.OnResponseStreamed -= OnBotResponseStreamed;
            }
        }

        void OnBotResponseStreamed(string response)
        {
            streamTokenBuffer.Add(response);

            // arbitrary number
            if (streamTokenBuffer.Count >= streamChunkSize)
            {
                string codes = string.Join("", streamTokenBuffer);
                streamTokenBuffer.Clear();

                decoder?.Decode(codes);
            }
        }

        void OnBotResponseGenerated(string response)
        {
            Debug.Log($"response: {response}");
            if (streamTokenBuffer.Count > 0)
            {
                string codes = string.Join("", streamTokenBuffer);
                streamTokenBuffer.Clear();

                decoder?.Decode(codes);
            }
        }
    }
}
