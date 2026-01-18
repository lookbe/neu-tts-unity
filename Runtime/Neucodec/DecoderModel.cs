using LlamaCpp;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace NeuTTS
{
    public class DecoderModel : BackgroundRunner
    {
        [Header("Model")]
        public string modelPath = string.Empty;

        // Define a delegate (or use Action<T>)
        public delegate void StatusChangedDelegate(ModelStatus status);
        public event StatusChangedDelegate OnStatusChanged;

        private ModelStatus _status = ModelStatus.Init;

        // Public getter, no public setter
        public ModelStatus status
        {
            get => _status;
            protected set
            {
                if (_status != value)
                {
                    _status = value;
                    OnStatusChanged?.Invoke(_status);
                }
            }
        }

        protected void PostStatus(ModelStatus newStatus)
        {
            unityContext?.Post(_ => status = newStatus, null);
        }

        async void OnDestroy()
        {
            await BackgroundStop();
            FreeModel();
        }

        // Define a delegate (or use Action<T>)
        public delegate void ResponseGeneratedDelegate(float[] response);
        public event ResponseGeneratedDelegate OnResponseGenerated;

        InferenceSession _session;

        public void InitModel()
        {
            if (string.IsNullOrEmpty(modelPath))
            {
                Debug.LogError("path not set");
                return;
            }

            if (_status != ModelStatus.Init)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Loading;
            RunBackground(RunInitModel);
        }

        void RunInitModel(CancellationToken cts)
        {
            try
            {
                Debug.Log($"Load model at {modelPath}");

                _session = new InferenceSession(modelPath);
                if (_session == null)
                {
                    throw new System.Exception("unable to load model");
                }

                Debug.Log("Load model done");
                PostStatus(ModelStatus.Ready);
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"An unexpected error occurred: {ex.Message}");

                FreeModel();
                PostStatus(ModelStatus.Init);
            }
        }

        private ulong _inputIndex = 0;
        private ulong _nextExpectedIndex = 0;
        private Dictionary<ulong, float[]> _responseCache = new Dictionary<ulong, float[]>();
        private Dictionary<ulong, Task> _backgroundTasks = new Dictionary<ulong, Task>();

        private class DecodePayload : IBackgroundPayload
        {
            public string Codes;
            public ulong Index;
        }

        public void Decode(string codes)
        {
            // harcoded value from onnx
            if (string.IsNullOrEmpty(codes))
            {
                Debug.LogWarning("wrong codes");
                return;
            }

            if (_session == null)
            {
                Debug.LogError("model not loaded");
                return;
            }

            if (status != ModelStatus.Ready && status != ModelStatus.Generate)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Generate;
            RunBackgroundDecode(codes);
        }

        protected void RunBackgroundDecode(string codes)
        {
            ulong index = _inputIndex++;
            _responseCache[index] = null;

            CancellationTokenSource cts = new CancellationTokenSource();
            DecodePayload payload = new DecodePayload() { Codes = codes, Index = index };

            var task = Task.Run(() =>
            {
                try
                {
                    RunDecode(payload, cts.Token);
                }
                catch (Exception ex)
                {
                    Debug.LogError($"Error in background task: {ex.Message}");
                }
            }, cts.Token);

            _backgroundTasks[index] = task;
        }

        void RunDecode(DecodePayload payload, CancellationToken cts)
        {
            try
            {
                float[] audioData = DecodeInternal(payload.Codes, cts);
                PostResponse(payload.Index, audioData);
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"An unexpected error occurred during RunDecode: {ex.Message}");
                PostResponse(payload.Index, new float[0]);
            }

        }

        float[] DecodeInternal(string codes, CancellationToken cts)
        {
            var matches = Regex.Matches(codes, @"<\|speech_(\d+)\|>");

            List<int> speechIds = new List<int>();
            foreach (Match match in matches)
            {
                if (int.TryParse(match.Groups[1].Value, out int id))
                {
                    speechIds.Add(id);
                }
            }

            if (speechIds.Count > 0)
            {
                int[] dimensions = new int[] { 1, 1, speechIds.Count };
                return DecodeCode(speechIds.ToArray(), dimensions, cts);
            }

            return Array.Empty<float>();
        }

        float[] DecodeCode(int[] codes, int[] dimensions, CancellationToken cts)
        {
            if (dimensions.Length != 3 || dimensions[1] != 1)
            {
                throw new ArgumentException("Codes should be of shape [B, 1, F].");
            }

            var inputTensor = new DenseTensor<int>(codes, dimensions);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("codes", inputTensor)
            };

            using (var results = _session.Run(inputs))
            {
                var output = results.First().AsTensor<float>();
                return output.ToArray();
            }
        }

        void FreeModel()
        {
            _session?.Dispose();
        }

        void PostResponse(ulong index, float[] response)
        {
            unityContext?.Post(_ => ProcessResponse(index, response), null);
        }

        // process this on main unity thread
        void ProcessResponse(ulong index, float[] response)
        {
            _backgroundTasks.Remove(index);
            _responseCache[index] = response;

            while (true)
            {
                _responseCache.TryGetValue(_nextExpectedIndex, out float[] nextResponse);
                if (nextResponse == null)
                {
                    break;
                }

                OnResponseGenerated?.Invoke(nextResponse);

                _responseCache.Remove(_nextExpectedIndex);
                _nextExpectedIndex++;
            }

            if (_responseCache.Count == 0)
            {
                status = ModelStatus.Ready;
            }
        }
    }
}
