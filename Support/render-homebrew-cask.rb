#!/usr/bin/env ruby
# frozen_string_literal: true

require "fileutils"

version, sha256, output = ARGV
abort "usage: Support/render-homebrew-cask.rb VERSION SHA256 OUTPUT" unless ARGV.length == 3
abort "invalid Lalia version: #{version}" unless version.match?(/\A(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\z/)
abort "invalid SHA-256: #{sha256}" unless sha256.match?(/\A[0-9a-f]{64}\z/)

FileUtils.mkdir_p(File.dirname(output))
File.write(output, <<~CASK)
  cask "lalia" do
    version "#{version}"
    sha256 "#{sha256}"

    url "https://github.com/cosgroveb/lalia/releases/download/v\#{version}/Lalia-\#{version}.dmg"
    name "Lalia"
    desc "Native macOS menu-bar voice dictation"
    homepage "https://github.com/cosgroveb/lalia"

    livecheck do
      url :url
      strategy :github_latest
    end

    depends_on arch: :arm64
    depends_on macos: ">= :tahoe"

    app "Lalia.app"
  end
CASK
