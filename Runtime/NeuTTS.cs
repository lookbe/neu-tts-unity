using LlamaCpp;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.IO;
using System.Text;
using UnityEngine;
using Newtonsoft.Json;

namespace NeuTTS
{
    public class NeuTTS : MonoBehaviour
    {
        public string neuttsModelPath = string.Empty;
        public string neucodecDecoderModelPath = string.Empty;

        [Header("Phonemizer")]
        public string phonemizerModelPath = string.Empty;
        public string phonemizerConfigPath = string.Empty;
        public string phonemizerDictPath = string.Empty;

        [Header("Reference Audio")]
        public string refAudioPath;
        public string refTranscriptPath;

        protected NeuTTSModel neutts;
        protected DecoderModel decoder;
        protected PhonemizerModel phonemizer;
        protected AudioSource audioSource;

        private Queue<float> audioQueue = new Queue<float>();

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

        public void InitModel()
        {
            if (string.IsNullOrEmpty(neuttsModelPath))
            {
                return;
            }

            if (string.IsNullOrEmpty(neucodecDecoderModelPath))
            {
                return;
            }

            if (string.IsNullOrEmpty(phonemizerModelPath) || string.IsNullOrEmpty(phonemizerConfigPath))
            {
                return;
            }

            if (_status != ModelStatus.Init)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Loading;
            StartCoroutine(RunInitModel());
        }

        IEnumerator RunInitModel()
        {
            Debug.Log($"Load neutts model");

            neutts.modelPath = neuttsModelPath;
            neutts.InitModel();

            decoder.modelPath = neucodecDecoderModelPath;
            decoder.InitModel();

            phonemizer.modelPath = phonemizerModelPath;
            phonemizer.configPath = phonemizerConfigPath;
            phonemizer.dictPath = phonemizerDictPath;
            phonemizer.InitModel();

            yield return new WaitWhile(() => neutts.status != ModelStatus.Ready);
            yield return new WaitWhile(() => decoder.status != ModelStatus.Ready);
            yield return new WaitWhile(() => phonemizer.status != ModelStatus.Ready);

            yield return StartCoroutine(EncodeRef());

            Debug.Log("Load model done");

            status = ModelStatus.Ready;
        }

        public void Prompt(string prompt)
        {
            if (string.IsNullOrEmpty(prompt))
            {
                Debug.LogWarning("invalid prompt");
                return;
            }

            if (status != ModelStatus.Ready)
            {
                Debug.LogError("invalid status");
                return;
            }

            status = ModelStatus.Generate;
            phonemizer.Phonemize(prompt);
            StartCoroutine(WaitForGenerationAndPlaybackDone());
        }

        IEnumerator WaitForGenerationAndPlaybackDone()
        {
            yield return new WaitUntil(() => phonemizer.status == ModelStatus.Ready);
            yield return new WaitUntil(() => neutts.status == ModelStatus.Ready);
            yield return new WaitUntil(() => decoder.status == ModelStatus.Ready);

            // Wait for all audio samples to be played
            yield return new WaitUntil(() => audioQueue.Count == 0);
            status = ModelStatus.Ready;
        }

        public void Stop()
        {
            if (status != ModelStatus.Generate)
            {
                Debug.Log("already stopped");
                return;
            }

            neutts.Stop();
        }

        // harcoded value from decoder
        private const int SampleRate = 24000;
        private const int Channels = 1;

        private void Awake()
        {
            neutts = GetComponentInChildren<NeuTTSModel>();
            decoder = GetComponentInChildren<DecoderModel>();
            phonemizer = GetComponentInChildren<PhonemizerModel>();
            audioSource = GetComponent<AudioSource>();

            audioSource.clip = AudioClip.Create("StreamingClip", SampleRate * 60, Channels, SampleRate, true, OnAudioRead);
            audioSource.loop = true;
            audioSource.Play();
        }

        private void OnEnable()
        {
            neutts.OnStatusChanged += OnModelStatusChanged;
            decoder.OnStatusChanged += OnModelStatusChanged;
            phonemizer.OnStatusChanged += OnModelStatusChanged;

            phonemizer.OnResponseGenerated += OnPhonemeResponse;
            decoder.OnResponseGenerated += OnResponseGenerated;
        }

        private void OnDisable()
        {
            neutts.OnStatusChanged -= OnModelStatusChanged;
            decoder.OnStatusChanged -= OnModelStatusChanged;
            phonemizer.OnStatusChanged -= OnModelStatusChanged;

            phonemizer.OnResponseGenerated -= OnPhonemeResponse;
            decoder.OnResponseGenerated -= OnResponseGenerated;
        }

        void OnModelStatusChanged(ModelStatus status)
        {
            if (status == ModelStatus.Error)
            {
                StopAllCoroutines();
                this.status = ModelStatus.Error;
            }
        }

        string transcript = string.Empty;
        string audioText = string.Empty;

        void OnPhonemeResponse(string phonemeString)
        {
            if (string.IsNullOrEmpty(transcript))
            {
                transcript = phonemeString;
            }
            else
            {
                Debug.Log($"prompt: {phonemeString}");
                neutts.PromptWithClone(phonemeString, transcript, audioText);
            }
        }

        void OnResponseGenerated(float[] audioChunk)
        {
            foreach (var s in audioChunk)
                audioQueue.Enqueue(s);
        }

        private void OnAudioRead(float[] data)
        {
            for (int i = 0; i < data.Length; i++)
            {
                if (audioQueue.Count > 0)
                    data[i] = audioQueue.Dequeue();
                else
                    data[i] = 0f;
            }
        }

        private IEnumerator EncodeRef()
        {
            var loadTask = Task.Run(() =>
            {
                string audioTokens = File.ReadAllText(refAudioPath);
                int[] referenceAudio = JsonConvert.DeserializeObject<int[]>(audioTokens);

                var sb = new StringBuilder(referenceAudio.Length * 16);
                foreach (int idx in referenceAudio)
                {
                    sb.Append("<|speech_").Append(idx).Append("|>");
                }
                audioText = sb.ToString();

                string transcriptText = File.ReadAllText(refTranscriptPath);
                phonemizer.Phonemize(transcriptText);
            });

            yield return new WaitUntil(() => loadTask.IsCompleted);
            yield return new WaitUntil(() => phonemizer.status == ModelStatus.Ready);
        }
    }
}
