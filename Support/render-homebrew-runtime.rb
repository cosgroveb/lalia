#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"

version, sha256, output = ARGV
abort "usage: Support/render-homebrew-runtime.rb VERSION SHA256 OUTPUT" unless ARGV.length == 3
abort "invalid Lalia version: #{version}" unless version.match?(/\A(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\z/)
abort "invalid SHA-256: #{sha256}" unless sha256.match?(/\A[0-9a-f]{64}\z/)

FileUtils.mkdir_p(File.dirname(output))
File.write(output, <<~FORMULA)
  class LaliaRuntime < Formula
    desc "Private local transcription runtime for Lalia"
    homepage "https://github.com/cosgroveb/lalia"
    url "https://github.com/cosgroveb/lalia/archive/refs/tags/v#{version}.tar.gz"
    version "#{version}"
    sha256 "#{sha256}"
    depends_on "cmake" => :build
    depends_on "rust" => :build
    depends_on arch: :arm64
    depends_on macos: :sequoia
    depends_on "python@3.11"

    resource "omlx" do
      url "https://github.com/jundot/omlx/archive/23b671f64f0a64f532e38d692a6c480ea704418d.tar.gz"
      sha256 "cd7ce61abf65c4b0fd2a0431dd3053a68e8efde21569ee2f8d30e3365e2709fc"
    end

    resource "mlx-audio" do
      url "https://github.com/cosgroveb/mlx-audio.git",
        revision: "67408c4910edd83935f5618780e51c403ded2676"
    end

    resource "mlx-vlm" do
      url "https://github.com/Blaizzy/mlx-vlm.git",
        revision: "9790231827a890eade1a626643a6bc94a1317c72"
    end

    resource "mlx-lm" do
      url "https://github.com/ml-explore/mlx-lm.git",
        revision: "94cdcae13b266c337bcaca09b97b9c5a9c0e2cde"
    end

    resource "mlx-embeddings" do
      url "https://github.com/Blaizzy/mlx-embeddings.git",
        revision: "32981fa4e8064ed664b52071789dd18271fe4206"
    end

    resource "cohere-melody-sdist" do
      url "https://files.pythonhosted.org/packages/62/ab/80ef37144fefcd54ffa14c8fee283fa607b40f4bd1c7ffe92f018f2a96f2/cohere_melody-0.15.3.tar.gz"
      sha256 "9cc13cca042243cbf76dbe7dc9345b0a7c6c25ad3a796caada0cdda56d425823"
    end

    resource "pydantic-core-sdist" do
      url "https://files.pythonhosted.org/packages/af/8e/23ec1021834e35041fbb2f10946141e6acdfd39379c45cc4d65a4e8a1d98/pydantic_core-2.50.2.tar.gz"
      sha256 "1b0e8dc3932cf4e9ed5727570325cacc716b9409ffc794a221c0ccc72e8eb775"
    end

    resource "rpds-py-sdist" do
      url "https://files.pythonhosted.org/packages/42/68/3bd46b8a5e01d3c2ebdf9c5e9497912e3fe0cde02bac21a7130ca866e403/rpds_py-2026.9.1.tar.gz"
      sha256 "4793ef7f78268b124b73fa933440f01d258bbae01de9fa53e9080c9ab0425a12"
    end

    resource "tiktoken-sdist" do
      url "https://files.pythonhosted.org/packages/66/62/167a842aa0429d45f5e797354fd4343a96f6043d67d0513c675c7b8d36e6/tiktoken-0.14.0.tar.gz"
      sha256 "231dec90efcdccf1b565a1416107736f1e09b1a08fe736ef9d6363e626d03874"
    end

    service do
      run [opt_bin/"lalia-runtime", "serve"]
      keep_alive true
      working_dir var
      log_path var/"log/lalia-runtime.log"
      error_log_path var/"log/lalia-runtime.log"
    end

    def install
      system "python3.11", "-m", "venv", libexec
      ENV.prepend_path "PATH", libexec/"bin"
      ENV.append "LDFLAGS", "-Wl,-headerpad_max_install_names"
      ENV.append "RUSTFLAGS", "-C link-arg=-Wl,-headerpad_max_install_names"
      system libexec/"bin/pip", "install", "--disable-pip-version-check",
             "--no-build-isolation", "--require-hashes",
             "-r", buildpath/"Support/runtime/requirements.lock"
      %w[cohere-melody-sdist pydantic-core-sdist rpds-py-sdist tiktoken-sdist].each do |name|
        resource(name).stage do
          system libexec/"bin/pip", "install", "--force-reinstall", "--no-deps",
                 "--no-build-isolation", "."
        end
      end
      %w[mlx-audio mlx-vlm mlx-lm mlx-embeddings].each do |name|
        resource(name).stage do
          system libexec/"bin/pip", "install", "--no-deps", "--no-build-isolation", "."
        end
      end
      resource("omlx").stage do
        system libexec/"bin/pip", "install", "--no-deps", "--no-build-isolation", "."
      end
      site_packages = Utils.safe_popen_read(libexec/"bin/python", "-c",
                                            "import sysconfig; print(sysconfig.get_paths()['purelib'])").strip
      system libexec/"bin/python", buildpath/"Support/runtime/patch-audio-usage.py",
             "--site-packages", site_packages,
             "--backup-dir", buildpath/"runtime-patch-backup"
      cp buildpath/"Support/runtime/lalia_runtime.py", libexec/"lalia_runtime.py"
      cp buildpath/"Support/runtime/patch-audio-usage.py", libexec/"patch-audio-usage.py"
      (bin/"lalia-runtime").write <<~SH
        #!/bin/bash
        exec "\#{libexec}/bin/python" "\#{libexec}/lalia_runtime.py" "$@"
      SH
      chmod 0755, bin/"lalia-runtime"
    end

    test do
      assert_match "usage: lalia-runtime", shell_output("\#{bin}/lalia-runtime --help 2>&1")
      system libexec/"bin/python", "-m", "pip", "check"
      system libexec/"bin/python", "-c", <<~PYTHON
        from mlx_audio.stt.models.moss_transcribe_diarize import Model
        from omlx.api.audio_models import AudioTranscriptionResponse
        assert Model and AudioTranscriptionResponse
      PYTHON
      assert_equal "0.7.2", shell_output(
        "\#{libexec}/bin/python -c 'from importlib.metadata import version; print(version(\\\"omlx\\\"))'",
      ).strip
      assert_equal "0.6.2", shell_output(
        "\#{libexec}/bin/python -c 'from importlib.metadata import version; print(version(\\\"mlx-audio\\\"))'",
      ).strip
    end
  end
FORMULA
